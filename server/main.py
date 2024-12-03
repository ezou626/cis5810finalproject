import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import cv2
import sys
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv('./.env', override=True)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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

# ANALYSIS_WINDOW = 60  # seconds
# CAPTION_GRANULARITY = 10

def extract_frames(video_url: str):
    """
    Extracts frames from a video URL or file path.
    """
    video_capture = cv2.VideoCapture(video_url)
    success, frame = video_capture.read()
    while success:
        yield frame
        success, frame = video_capture.read()
    video_capture.release()

def batch_frames(frames, batch_size):
    """
    Groups frames into batches of a specified size.
    """
    batch = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def generate_caption_with_blip(image: np.ndarray) -> str:
    """
    Generate a caption for a single image using BLIP.

    Args:
        image (np.ndarray): The image as a NumPy array.

    Returns:
        str: The generated caption.
    """
    # Convert NumPy image to PIL format
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Preprocess the image and generate the caption
    inputs = processor(images=image_pil, return_tensors="pt")
    outputs = model.generate(**inputs)

    # Decode and return the caption
    return processor.decode(outputs[0], skip_special_tokens=True)

async def caption_images_with_blip(images: list[np.ndarray]) -> list[str]:
    """
    Generate captions for a batch of images using BLIP.

    Args:
        images (list[np.ndarray]): A list of images as NumPy arrays.

    Returns:
        list[str]: A list of generated captions.
    """
    captions = []
    for image in images:
        caption = generate_caption_with_blip(image)
        captions.append(caption)
    return captions

@app.get('/captions')
async def get_captions(video_url: str):
    """
    Extracts frames from a video, processes them in batches, and generates captions using BLIP.
    """
    async def caption_stream():
        try:
            frames = extract_frames(video_url)  # Extract frames from the video
            for frame_batch in batch_frames(frames, batch_size=5):  # Batch the frames
                captions = await caption_images_with_blip(frame_batch)  # Generate captions
                for caption in captions:
                    yield f"data: {caption}\n\n"  # Stream captions
                    await asyncio.sleep(1)  # Control the streaming rate
        except Exception as e:
            print(e, file=sys.stderr)
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(caption_stream(), media_type="text/event-stream")

@app.get('/captions_debug')
async def get_captions_debug():
    """
    Debug endpoint to simulate captions stream for testing purposes.
    """
    index = 0

    async def caption_stream():
        nonlocal index
        while True:
            await asyncio.sleep(1)
            index += 1
            yield f"data: Debug Message {index}\n\n"

    return StreamingResponse(caption_stream(), media_type="text/event-stream")
