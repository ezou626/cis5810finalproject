import numpy as np
import os, time, sys
import asyncio
import httpx
import base64
import cv2
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

ANALYSIS_WINDOW = 60  # seconds
CAPTION_GRANULARITY = 10

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
    index = 0
    for image in images:
        if index % ANALYSIS_WINDOW == 0:
            caption = generate_caption_with_blip(image)
            captions.append(caption)
    return captions


def extract_frames(video_url):
    video_capture = cv2.VideoCapture(video_url)
    success, frame = video_capture.read()
    while success:
        yield frame
        success, frame = video_capture.read()
    video_capture.release()

def batch_frames(frames, batch_size):
    batch = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
