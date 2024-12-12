import asyncio
import io
import sys
import time
from uvicorn import Server, Config
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
from dotenv import load_dotenv
from commentary import extract_frames, batch_frames, caption_images_with_gemini
import google.generativeai as genai
from contextlib import asynccontextmanager
from models import setup_models, get_yolo_model
from zoom import get_image_zoom_box, get_zoomed_frame
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

yolo_pool = ProcessPoolExecutor(max_workers=4)
transform_pool = ProcessPoolExecutor(max_workers=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv('.env', override=True)
    setup_models()
    yield

# FastAPI app setup
app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

ANALYSIS_WINDOW = 5  # seconds

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/play_video_mod')
async def play_video_mod(url: str, time: int):
    """
    Streams video content from a given URL with YOLO object detection and auto-zoom.
    Maintains aspect ratio and frame size.
    """

    yolo_model = get_yolo_model()
    task_queue = asyncio.Queue(maxsize=10)
    box_queue = asyncio.Queue(maxsize=10)

    async def worker():
        while True:
            if not box_queue.empty():
                await asyncio.sleep(0.1)
                continue
            if task_queue.empty():
                await asyncio.sleep(0.1)
                continue
            try:
                task = task_queue.get_nowait()
                result = await task
                box_queue.put_nowait(result)
            except Exception as e:
                print(f"Error processing frame: {e}", file=sys.stderr)
            await asyncio.sleep(0.5)

    workers = [asyncio.create_task(worker()) for _ in range(1)]

    async def process_video():
        cap = cv2.VideoCapture(url)
        framerate = round(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        loop = asyncio.get_event_loop()
        index = -1
        current_box = (0, 0, width, height)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                index += 1
                index %= framerate
                
                if index == 0:
                    # submit this frame for processing
                    try:
                        task_queue.put_nowait(
                            loop.run_in_executor(
                                yolo_pool, get_image_zoom_box, frame, width, height, yolo_model
                            )
                        )
                    except asyncio.QueueFull as e:
                        print(f"Queue is full: {e}", file=sys.stderr)

                # get the next box if available
                while not box_queue.empty():
                    current_box = box_queue.get_nowait()
                    
                img_bytes = await loop.run_in_executor(transform_pool, get_zoomed_frame, frame, width, height, *current_box)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        img_bytes + b'\r\n')

        except Exception as e:
            print(f"Error during video processing: {e}", file=sys.stderr)
        finally:
            cap.release()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get('/captions')
async def get_captions(video_url: str):
    """Extract frames and generate captions."""
    framerate, width, height = get_video_metadata(video_url)
    if width * height > 480 * 480:
        scale = (480 * 480) / (width * height)
        width = int(width * scale)
        height = int(height * scale)

    async def caption_stream():
        captions = []
        try:
            frames = extract_frames(video_url)
            async for frame_batch in batch_frames(frames, batch_size=ANALYSIS_WINDOW, framerate=framerate):
                print('got a batch', file=sys.stderr)
                start = time.time()
                caption = await caption_images_with_gemini(frame_batch, width, height, captions)
                captions.append(caption)
                end = time.time()
                yield f"data: {caption}\n\n"
                # await asyncio.sleep(max(0, ANALYSIS_WINDOW - (end - start)))
        except Exception as e:
            print(f"Error during caption generation: {e}", file=sys.stderr)
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(caption_stream(), media_type="text/event-stream")

def get_video_metadata(video_url: str) -> tuple[int, int, int]:
    """Get video metadata from a given URL."""
    cap = cv2.VideoCapture(video_url)
    framerate = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return framerate, width, height

if os.environ.get('MACHINE_TYPE', 'not') == 'windows':
    loop = asyncio.new_event_loop()
    config = Config(app=app)
    server = Server(config)
    loop.run_until_complete(server.serve())

from fastapi.responses import FileResponse

@app.get('/download_processed_video')
async def download_processed_video():
    # Replace 'processed_video.mp4' with the actual path to the processed video file
    file_path = "processed_video.mp4"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed video not found.")

    return FileResponse(file_path, media_type="video/mp4", filename="processed_video.mp4")
