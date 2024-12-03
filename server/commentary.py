import numpy as np
import os, time, sys
import asyncio
import httpx
import base64
import cv2
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import google.generativeai as genai
gemini_model = genai.GenerativeModel(model_name = "gemini-1.5-flash")

def preprocess_image(image, width: int, height: int):
    image = cv2.resize(image, (width, height))
    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_frames(video_url):
    video_capture = cv2.VideoCapture(video_url)
    success, frame = video_capture.read()
    while success:
        yield frame
        success, frame = video_capture.read()
    video_capture.release()

def batch_frames(frames, batch_size, framerate):
    batch = []
    for i, frame in enumerate(frames):
        if i % framerate == 0:
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def generate_caption_with_blip(image: np.ndarray, width: int, height: int) -> str:
    """
    Generate a caption for a single image using BLIP.

    Args:
        image (np.ndarray): The image as a NumPy array.

    Returns:
        str: The generated caption.
    """

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height), Image.Resampling.LANCZOS)

    # Preprocess the image and generate the caption
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)

    # Decode and return the caption
    return processor.decode(outputs[0], skip_special_tokens=True)

def caption_images_with_blip(images, width: int, height: int):
    """
    Generate captions for a batch of images using BLIP.

    Args:
        images (list[np.ndarray]): A list of images as NumPy arrays.

    Returns:
        list[str]: A list of generated captions.
    """
    for image in images:
        caption = generate_caption_with_blip(image, width, height)
        yield caption

async def caption_images_with_gemini(images, width: int, height: int):
    instruction = "Describe the sequence occuring the in these images in a single detailed sentence. Do not include any other words in your response besides the sentence."
    prompt = [
        {
            'mime_type':'image/jpeg', 
            'data': base64.b64encode(
                        cv2.imencode('.jpeg', 
                                     preprocess_image(image, width, height)
                    )[1]).decode()
        } for image in images
    ]
    prompt.append(instruction)
    response = await gemini_model.generate_content_async(prompt)
    return response.text.strip()

async def get_caption_for_batch(images, width: int, height: int):
    """
    Generate caption for a batch of images

    Args:
        images (list[np.ndarray]): A list of images as NumPy arrays.

    Returns:
        list[str]: A list of generated captions.
    """
    return await caption_images_with_gemini(images, width, height)