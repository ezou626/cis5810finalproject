import json
import asyncio
import httpx
import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv('server/.env', override=True)

async def run_inference():
    # Define the URL and headers
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": os.environ.get("U_API_KEY")}
    data = {
        "model": "https://hub.ultralytics.com/models/5NOJhLhigIjjuOABqyUQ",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.45,
    }
    
    image_path = "server/test_frame.jpg"
    output_path = "server/output_frame.png"

    # Read the image file asynchronously
    async with httpx.AsyncClient() as client:
        with open(image_path, "rb") as f:
            files = {"file": ("image.jpg", f, "image/jpg")}
            # Send the POST request asynchronously
            response = await client.post(url, 
                                         headers=headers, 
                                         data=data, 
                                         files=files,
                                         timeout=30.0)
    
        # Check for successful response
        response.raise_for_status()
        predictions = response.json()  # Extract predictions

    # Print inference results for inspection
    print(json.dumps(predictions, indent=2))

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Draw bounding boxes
    for result in predictions['images'][0]['results']:
        box = result['box']
        confidence = result['confidence']
        label = result['name']
        
        # Extract points
        points = np.array([
            [box['x1'], box['y1']],
            [box['x2'], box['y2']],
        ], np.int32)

        # Draw the quadrilateral
        cv2.rectangle(image, points[0], points[1], (0, 255, 0), 2)

        # Add label and confidence score at the top-left point
        x_text, y_text = int(box['x1']), int(box['y1'])
        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x_text, y_text - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Save the result image
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

# Run the async function
asyncio.run(run_inference())