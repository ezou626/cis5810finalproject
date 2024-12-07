import asyncio
import io
import time
from uvicorn import Server, Config
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from pydantic import BaseModel
from commentary import extract_frames, batch_frames, get_caption_for_batch
import google.generativeai as genai

# Load environment variables
load_dotenv('.env', override=True)
genai.configure(api_key=os.environ.get('G_API_KEY'))

# FastAPI app setup
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global variables
user_selected_box = None  # Store the user-defined bounding box
yolo_model = None  # YOLO model placeholder

# Load YOLO model
def load_model():
    global yolo_model
    try:
        yolo_model = YOLO("yolov5s.pt")  # Load pre-trained YOLOv5 model
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        yolo_model = None

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

@app.post('/set_tracking_box')
def set_tracking_box(box: BoundingBox):
    """Set the bounding box to track."""
    global user_selected_box
    user_selected_box = box.dict()
    return {"message": "Tracking box set successfully"}

@app.get('/get_first_frame')
async def get_first_frame(url: str):
    """Extract and serve the first frame of the video."""
    try:
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise HTTPException(status_code=400, detail="Unable to read the first frame. Check the video URL.")

        _, buffer = cv2.imencode('.jpeg', frame)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/play_video_mod')
async def play_video_mod(url: str):
    """
    Streams video content from a given URL with YOLO object detection and auto-zoom.
    Maintains aspect ratio and frame size.
    """
    async def process_video():
        cap = cv2.VideoCapture(url)
        framerate = round(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO detection
                results = yolo_model(frame)

                # Parse detections
                detections = []
                for r in results[0].boxes:
                    box = r.xyxy.numpy().tolist()[0]  # Bounding box in [x1, y1, x2, y2]
                    conf = r.conf.item()  # Confidence score
                    cls = r.cls.item()  # Class ID
                    if conf > 0.5 and cls == 0:  # Only include high-confidence "person" detections
                        detections.append({"box": box, "conf": conf, "class": cls})
                ################################################################################################
                #MATH IS HERE
                ################################################################################################
                if detections:
                    # Select the largest object for zooming
                    largest_detection = max(detections, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
                    x1, y1, x2, y2 = map(int, largest_detection["box"])

                    # Calculate the bounding box dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    original_aspect_ratio = width / height

                    # Determine the size of the sliding frame (scaled as large as possible)
                    if box_width / box_height > original_aspect_ratio:
                        # Wider than original aspect ratio: fit width
                        frame_width = box_width
                        frame_height = int(box_width / original_aspect_ratio)
                    else:
                        # Taller than original aspect ratio: fit height
                        frame_height = box_height
                        frame_width = int(box_height * original_aspect_ratio)

                    # Ensure the frame fits within the image bounds
                    frame_width = min(frame_width, width)
                    frame_height = min(frame_height, height)

                    # Center the frame on the bounding box
                    center_x = x1 + box_width // 2
                    center_y = y1 + box_height // 2
                    frame_x1 = max(0, center_x - frame_width // 2)
                    frame_y1 = max(0, center_y - frame_height // 2)
                    frame_x2 = min(width, frame_x1 + frame_width)
                    frame_y2 = min(height, frame_y1 + frame_height)

                    # Ensure the frame remains within bounds
                    frame_x1 = max(0, frame_x2 - frame_width)
                    frame_y1 = max(0, frame_y2 - frame_height)

                    # Extract the content within the frame
                    sliding_frame = frame[frame_y1:frame_y2, frame_x1:frame_x2]

                    # Resize the sliding frame to the screen size
                    zoomed_frame = cv2.resize(sliding_frame, (width, height))

                ################################################################################################################################
                ################################################################################################################################

                    # Encode and yield the frame
                    _, buffer = cv2.imencode('.jpeg', zoomed_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
                else:
                    # No detections, yield the original frame
                    _, buffer = cv2.imencode('.jpeg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')

                await asyncio.sleep(1 / framerate)

        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            cap.release()

    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")

ANALYSIS_WINDOW = 10  # seconds

@app.get('/captions')
async def get_captions(video_url: str):
    """Extract frames and generate captions using BLIP."""
    framerate, width, height = get_video_metadata(video_url)
    if width * height > 480 * 480:
        scale = (480 * 480) / (width * height)
        width = int(width * scale)
        height = int(height * scale)

    async def caption_stream():
        try:
            frames = extract_frames(video_url)
            for frame_batch in batch_frames(frames, batch_size=ANALYSIS_WINDOW, framerate=framerate):
                start = time.time()
                caption = await get_caption_for_batch(frame_batch, width, height)
                end = time.time()
                yield f"data: {caption}\n\n"
                await asyncio.sleep(max(0, ANALYSIS_WINDOW - (end - start)))
        except Exception as e:
            print(f"Error during caption generation: {e}")
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
