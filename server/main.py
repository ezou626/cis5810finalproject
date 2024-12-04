import asyncio
from asyncio import new_event_loop
import base64
from uvicorn import Server, Config
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import sys, time
import os
import cv2
from dotenv import load_dotenv
from commentary import extract_frames, batch_frames, get_caption_for_batch
import google.generativeai as genai
from typing import List, Dict
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# Load environment variables
load_dotenv('.env', override=True)
genai.configure(api_key=os.environ.get('G_API_KEY'))

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def get_video_metadata(video_url: str) -> tuple[int, int, int]:
    """Get video metadata from a given URL

    Args:
        video_url (str): a URL to a video stream

    Returns:
        tuple[int, int, int]: a tuple containing the framerate, width, and height of the video stream
    """
    cap = cv2.VideoCapture(video_url)
    framerate = round(cap.get(5))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return framerate, width, height

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/play_video')
async def play_video(url: str):
    """
    Streams video content from a given URL.
    """
    async def iterfile():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(iterfile(), media_type="video/mp4")

detection_model = None

def load_model():
    global detection_model
    # Load the model from TensorFlow Hub
    detection_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

@app.on_event("startup")
def startup_event():
    load_model()

def detect_objects(frame: np.ndarray) -> List[Dict]:
    """Run object detection on a single frame."""
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_model(input_tensor)

    # Extract detection details
    boxes = detections["detection_boxes"][0].numpy()
    classes = detections["detection_classes"][0].numpy().astype(int)
    scores = detections["detection_scores"][0].numpy()

    # Filter detections by a score threshold
    threshold = 0.5
    valid_detections = [
        {"box": box, "class": cls, "score": score}
        for box, cls, score in zip(boxes, classes, scores) if score >= threshold
    ]
    return valid_detections

@app.get('/play_video_mod')
async def play_video_mod(url: str):
    """
    Streams video content from a given URL with object detection and auto-zoom.
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

                    # Crop and resize the frame to simulate zoom
                    zoomed_frame = frame[y1:y2, x1:x2]
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
            print(e, file=sys.stderr)

        finally:
            cap.release()

    return StreamingResponse(process_video(), media_type="multipart/x-mixed-replace; boundary=frame")


ANALYSIS_WINDOW = 10  # seconds

@app.get('/captions')
async def get_captions(video_url: str):
    """
    Extracts frames from a video, processes them in batches, and generates captions using BLIP.
    """
    framerate, width, height = get_video_metadata(video_url)
    if width * height > 480*480:
        scale = (480*480) / (width * height)
        width = int(width * scale)
        height = int(height * scale)
    async def caption_stream():
        try:
            frames = extract_frames(video_url)  # Extract frames from the video
            for frame_batch in batch_frames(frames, batch_size=ANALYSIS_WINDOW, framerate=framerate):  # Batch the frames
                start = time.time()
                caption = await get_caption_for_batch(frame_batch, width, height)
                end = time.time()
                yield f"data: {caption}\n\n"
                # consistent timing
                await asyncio.sleep(max(0, ANALYSIS_WINDOW - (end - start)))
        except Exception as e:
            print(e, file=sys.stderr)
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(caption_stream(), media_type="text/event-stream")

if os.environ.get('MACHINE_TYPE', 'not') == 'windows':
    loop = new_event_loop()
    config = Config(app=app)
    server = Server(config)
    loop.run_until_complete(server.serve())
