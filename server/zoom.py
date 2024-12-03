import asyncio
import httpx
import os
import cv2
import numpy as np

async def get_image_zoom_box(frame):
    # Define the URL and headers
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": os.environ.get("U_API_KEY")}
    data = {
        "model": "https://hub.ultralytics.com/models/5NOJhLhigIjjuOABqyUQ",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.45,
    }
    frame = cv2.imencode('.jpeg', frame)[1]

    # Read the image file asynchronously
    async with httpx.AsyncClient() as client:
        files = {"file": ("image.jpeg", frame.tobytes(), "image/jpeg")}
        # Send the POST request asynchronously
        response = await client.post(url, 
                                        headers=headers, 
                                        data=data, 
                                        files=files,
                                        timeout=30.0)
    
        # Check for successful response
        response.raise_for_status()
        predictions = response.json()  # Extract predictions

    # Draw bounding boxes
    x1, y1, x2, y2 = None
    for result in predictions['images'][0]['results']:
        box = result['box']
        
        # Extract points
    return x1, y1, x2, y2