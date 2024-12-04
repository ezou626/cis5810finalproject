import asyncio
from asyncio import new_event_loop
import base64
import io
from uvicorn import Server, Config
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sys, time
import os
import cv2
from dotenv import load_dotenv
from typing import List, Dict
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from commentary import extract_frames, batch_frames, get_caption_for_batch
import google.generativeai as genai
from pydantic import BaseModel


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
detection_model = None  # YOLO model placeholder

# Load TensorFlow Hub model
def load_model():
    global detection_model
    try:
        detection_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        detection_model = None

@app.on_event("startup")
def startup_event():
    load_model()

def get_video_metadata(video_url: str) -> tuple[int, int, int]:
    """Get video metadata from a given URL."""
    cap = cv2.VideoCapture(video_url)
    framerate = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return framerate, width, height

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
    user_selected_box = box.dict()  # Convert Pydantic model to dictionary
    return {"message": "Tracking box set successfully"}

@app.get('/get_first_frame')
async def get_first_frame(url: str):
    """Extract and serve the first frame of the video."""
    try:
        cap = cv2.VideoCapture(url)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"error": "Unable to read the first frame. Check the video URL."}

        _, buffer = cv2.imencode('.jpeg', frame)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except Exception as e:
        print(f"Error extracting first frame: {e}", file=sys.stderr)
        return {"error": str(e)}

def detect_objects(frame):
    """Run YOLO detection on a frame and return object bounding boxes."""
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_model(input_tensor)

    # Extract bounding box, class, and confidence score
    boxes = detections["detection_boxes"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)
    scores = detections["detection_scores"][0].numpy()

    # Filter by confidence threshold
    threshold = 0.5
    results = []
    for box, cls, score in zip(boxes, classes, scores):
        if score >= threshold:
            ymin, xmin, ymax, xmax = box
            results.append({
                "box": (xmin, ymin, xmax, ymax),  # Normalized coordinates
                "class": cls,
                "score": score
            })
    return results

def detect_objects_within_box(frame, user_box, detections):
    """Filter YOLO detections to focus on the user-defined bounding box."""
    x1, y1, x2, y2 = user_box
    filtered_detections = []

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (
        int(x1 * frame_width), int(y1 * frame_height),
        int(x2 * frame_width), int(y2 * frame_height)
    )

    for detection in detections:
        obj_x1, obj_y1, obj_x2, obj_y2 = detection["box"]

        # Convert normalized coordinates to pixel values
        obj_x1, obj_x2 = int(obj_x1 * frame_width), int(obj_x2 * frame_width)
        obj_y1, obj_y2 = int(obj_y1 * frame_height), int(obj_y2 * frame_height)

        # Check if the object's bounding box intersects with the user-defined box
        if not (obj_x2 < x1 or obj_x1 > x2 or obj_y2 < y1 or obj_y1 > y2):
            filtered_detections.append(detection)

    return filtered_detections

@app.get('/play_video_mod')
async def play_video_mod(url: str):
    """
    Streams video content from a given URL with object detection and auto-zoom.
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

                # Run object detection
                detections = detect_objects(frame)
                if detections:
                    # Select the largest object for zooming
                    largest_detection = max(detections, key=lambda d: d["box"][2] * d["box"][3])
                    box = largest_detection["box"]

                    # Convert box coordinates to pixel values
                    ymin, xmin, ymax, xmax = box
                    (x1, y1, x2, y2) = (int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height))

                    # Crop the frame to the area of interest
                    cropped_frame = frame[y1:y2, x1:x2]

                    # Calculate the aspect ratio of the cropped frame
                    cropped_height, cropped_width = cropped_frame.shape[:2]
                    original_aspect_ratio = width / height
                    cropped_aspect_ratio = cropped_width / cropped_height

                    # Adjust the crop to maintain the original aspect ratio
                    if cropped_aspect_ratio > original_aspect_ratio:
                        # Wider than original aspect ratio: adjust height
                        new_height = int(cropped_width / original_aspect_ratio)
                        delta_height = (new_height - cropped_height) // 2
                        y1 = max(0, y1 - delta_height)
                        y2 = min(height, y2 + delta_height)
                    elif cropped_aspect_ratio < original_aspect_ratio:
                        # Taller than original aspect ratio: adjust width
                        new_width = int(cropped_height * original_aspect_ratio)
                        delta_width = (new_width - cropped_width) // 2
                        x1 = max(0, x1 - delta_width)
                        x2 = min(width, x2 + delta_width)

                    # Ensure the adjusted crop is within frame bounds
                    y1, y2 = max(0, y1), min(height, y2)
                    x1, x2 = max(0, x1), min(width, x2)

                    # Final cropped frame
                    zoomed_frame = frame[y1:y2, x1:x2]

                    # Resize the cropped frame to the original dimensions
                    zoomed_frame = cv2.resize(zoomed_frame, (width, height))

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
            print(f"Error during video processing: {e}", file=sys.stderr)

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
            print(f"Error during caption generation: {e}", file=sys.stderr)
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(caption_stream(), media_type="text/event-stream")

if os.environ.get('MACHINE_TYPE', 'not') == 'windows':
    loop = new_event_loop()
    config = Config(app=app)
    server = Server(config)
    loop.run_until_complete(server.serve())
