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
from commentary import extract_frames, batch_frames, caption_images_with_gemini, caption_images_with_gemini_sync
from contextlib import asynccontextmanager
from models import setup_models, get_yolo_model, get_google_client
from zoom import get_image_zoom_box, get_zoomed_frame
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Debug helper function to ensure stderr messages are displayed immediately
def debug_print(message, file=sys.stderr):
    print(message, file=file, flush=True)

yolo_pool = ProcessPoolExecutor(max_workers=4)
transform_pool = ThreadPoolExecutor(max_workers=2)
caption_pool = ThreadPoolExecutor(max_workers=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    debug_print("[LIFESPAN] Starting application setup")
    load_dotenv('.env', override=True)
    setup_models()
    debug_print("[LIFESPAN] Models setup complete")
    # if os.environ.get('MACHINE_TYPE', 'not') == 'windows':
    #     loop = asyncio.new_event_loop()
    #     config = Config(app=app)
    #     server = Server(config)
    #     loop.run_until_complete(server.serve())
    yield
    debug_print("[LIFESPAN] Cleaning up application resources")

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

def index_wrapper(index, frame, width, height, yolo_model):
    debug_print(f"[YOLO] Processing frame {index}")
    return index, get_image_zoom_box(frame, width, height, yolo_model)

@app.get('/play_video_mod')
async def play_video_mod(url: str, imgKey: int):
    """
    Streams video content from a given URL with YOLO object detection and auto-zoom.
    Maintains aspect ratio and frame size.
    """
    debug_print(f"[VIDEO] Starting video processing for URL: {url}")

    cap = cv2.VideoCapture(url)
    framerate = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    debug_print(f"[VIDEO] Video metadata - FPS: {framerate}, Width: {width}, Height: {height}")

    yolo_model = get_yolo_model()
    task_queue = asyncio.Queue(maxsize=10)
    box_queue = asyncio.Queue(maxsize=10)
    frame_queue = asyncio.Queue(maxsize=framerate + 2) # only allow ~1 second ahead

    async def worker():
        worker_id = id(asyncio.current_task())
        debug_print(f"[WORKER {worker_id}] Starting worker")
        while True:
            try:
                if task_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                task = await task_queue.get()
                debug_print(f"[WORKER {worker_id}] Picking up task from queue")
                
                result = await task
                debug_print(f"[WORKER {worker_id}] Task completed, adding to box queue")
                
                await box_queue.put(result)
                task_queue.task_done()
            except asyncio.CancelledError:
                debug_print(f"[WORKER {worker_id}] Worker cancelled")
                break
            except Exception as e:
                debug_print(f"[WORKER {worker_id}] Error processing frame: {e}")

    workers = [asyncio.create_task(worker()) for _ in range(2)]

    async def populate_queues():
        debug_print("[POPULATE] Starting queue population")
        loop = asyncio.get_event_loop()
        index = -1

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    debug_print("[POPULATE] No more frames to read")
                    break

                index += 1
                
                if index % framerate == 0:
                    debug_print(f"[POPULATE] Processing frame {index}")
                    try:
                        if task_queue.full():
                            debug_print("[POPULATE] Task queue full, removing oldest task")
                            task_queue.get_nowait()
                        
                        task = loop.run_in_executor(
                            yolo_pool, index_wrapper, index, frame, width, height, yolo_model
                        )
                        await task_queue.put(task)
                    except Exception as e:
                        debug_print(f"[POPULATE] Error queueing task: {e}")

                try:
                    while frame_queue.full():
                        debug_print("[POPULATE] Frame queue full, waiting")
                        await asyncio.sleep(0.1)
                    frame_queue.put_nowait((index, frame))
                except Exception as e:
                    debug_print(f"[POPULATE] Error adding frame to queue: {e}")

        except Exception as e:
            debug_print(f"[POPULATE] Error during video processing: {e}")
        finally:
            cap.release()
            debug_print("[POPULATE] Video capture released")

    asyncio.create_task(populate_queues())

    async def process_video():
        debug_print("[PROCESS] Starting video processing")
        loop = asyncio.get_event_loop()
        current_box = (0, 0, width, height)
        box_index = -1

        try:
            while True:
                start = time.time()

                if frame_queue.empty():
                    await asyncio.sleep(0.1)
                    continue

                frame_index, frame = await frame_queue.get()
                debug_print(f"[PROCESS] Processing frame {frame_index}")

                while box_index < frame_index:
                    if not box_queue.empty():
                        box_index, current_box = await box_queue.get()
                        debug_print(f"[PROCESS] Got new bounding box for frame {box_index}")
                    else:
                        await asyncio.sleep(0.1)

                img_bytes = await loop.run_in_executor(transform_pool, get_zoomed_frame, frame, width, height, *current_box)
                debug_print(f"[PROCESS] Generated image bytes for frame {frame_index}")
                
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        img_bytes + b'\r\n')
                
                end = time.time()
                debug_print(f"[PROCESS] Frame processing time: {end - start}")
                await asyncio.sleep(max(0, 1 / framerate - (end - start)))

        except Exception as e:
            debug_print(f"[PROCESS] Error during video processing: {e}")
        finally:
            cap.release()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            debug_print("[PROCESS] Cleanup completed")

    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get('/captions')
async def get_captions(video_url: str):
    """Extract frames and generate captions."""
    debug_print(f"[CAPTIONS] Starting caption generation for {video_url}")
    
    framerate, width, height = get_video_metadata(video_url)
    debug_print(f"[CAPTIONS] Video metadata - FPS: {framerate}, Width: {width}, Height: {height}")
    
    if width * height > 480 * 480:
        scale = (480 * 480) / (width * height)
        width = int(width * scale)
        height = int(height * scale)
        debug_print(f"[CAPTIONS] Scaled down to Width: {width}, Height: {height}")

    loop = asyncio.get_event_loop()
    google_client = get_google_client()

    async def caption_stream():
        captions = []
        try:
            frames = extract_frames(video_url)
            
            start = time.time()
            async for frame_batch in batch_frames(frames, batch_size=ANALYSIS_WINDOW, framerate=framerate):
                debug_print(f'[CAPTION_STREAM] Processing batch of {len(frame_batch)} frames')
                
                caption = await loop.run_in_executor(
                    caption_pool, 
                    caption_images_with_gemini_sync, 
                    frame_batch, width, height, captions, google_client
                )
                debug_print(f'[CAPTION_STREAM] Generated caption: {caption}')
                
                captions.append(caption)
                end = time.time()
                
                yield f"data: {caption}\n\n"
                
                await asyncio.sleep(max(0, ANALYSIS_WINDOW - (end - start)))
                start = time.time()
        
        except Exception as e:
            debug_print(f"[CAPTION_STREAM] Error during caption generation: {e}")
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(caption_stream(), media_type="text/event-stream")

def get_video_metadata(video_url: str) -> tuple[int, int, int]:
    """Get video metadata from a given URL."""
    debug_print(f"[METADATA] Extracting metadata for {video_url}")
    cap = cv2.VideoCapture(video_url)
    framerate = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    debug_print(f"[METADATA] FPS: {framerate}, Width: {width}, Height: {height}")
    return framerate, width, height

from fastapi.responses import FileResponse

@app.get('/download_processed_video')
async def download_processed_video():
    # Replace 'processed_video.mp4' with the actual path to the processed video file
    file_path = "processed_video.mp4"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed video not found.")

    return FileResponse(file_path, media_type="video/mp4", filename="processed_video.mp4")
