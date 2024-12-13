import asyncio
import httpx
import os
import cv2
import numpy as np

async def get_image_zoom_box_net(frame):
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

def get_image_zoom_box(frame, width, height, yolo_model):
    results = yolo_model(frame)

    detections = []
    for r in results[0].boxes:
        box = r.xyxy.numpy().tolist()[0]  # Bounding box in [x1, y1, x2, y2]
        conf = r.conf.item()  # Confidence score
        cls = r.cls.item()  # Class ID
        
        if conf > 0.3 and cls == 0:
            area = (box[2] - box[0]) * (box[3] - box[1])
            detections.append({
                "box": box, 
                "conf": conf, 
                "class": cls,
                "area": area,
                "conf_area": conf * area
            })

    if not detections:
        return (0, 0, width, height)
    
    # # Get top 5 closest to centroid of bounding boxes
    # # get mean of bounding boxes
    # cx, cy = sum(d["box"][i] for d in detections for i in (0, 2)) / (2 * len(detections)), sum(d["box"][i] for d in detections for i in (1, 3)) / (2 * len(detections))
    # best_detections = sorted(detections, key=lambda d: (d["box"][0] + d["box"][2] - 2 * cx) ** 2 + (d["box"][1] + d["box"][3] - 2 * cy) ** 2)[:5]
    
    # # Get top 5 by confidence * area
    # best_detections = sorted(detections, key=lambda d: d["conf_area"], reverse=True)[:5]

    # Get top 5 by closeness to median
    cx, cy = np.median([d["box"][0] + d["box"][2] for d in detections]) / 2, np.median([d["box"][1] + d["box"][3] for d in detections]) / 2
    best_detections = sorted(detections, key=lambda d: (d["box"][0] + d["box"][2] - 2 * cx) ** 2 + (d["box"][1] + d["box"][3] - 2 * cy) ** 2)[:max(len(detections) // 2, 1)]

    # Find the smallest box that includes all detected boxes
    x1 = int(min(d["box"][0] for d in best_detections))
    y1 = int(min(d["box"][1] for d in best_detections))
    x2 = int(max(d["box"][2] for d in best_detections))
    y2 = int(max(d["box"][3] for d in best_detections))

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

    # Center the frame on the combined bounding box
    center_x = x1 + box_width // 2
    center_y = y1 + box_height // 2
    frame_x1 = max(0, center_x - frame_width // 2)
    frame_y1 = max(0, center_y - frame_height // 2)
    frame_x2 = min(width, frame_x1 + frame_width)
    frame_y2 = min(height, frame_y1 + frame_height)
    frame_x1 = max(0, frame_x2 - frame_width)
    frame_y1 = max(0, frame_y2 - frame_height)

    # Add a 10% buffer zone
    buffer_x = int(0.1 * (frame_x2 - frame_x1))
    buffer_y = int(0.1 * (frame_y2 - frame_y1))
    frame_x1 = max(0, frame_x1 - buffer_x)
    frame_y1 = max(0, frame_y1 - buffer_y)
    frame_x2 = min(width, frame_x2 + buffer_x)
    frame_y2 = min(height, frame_y2 + buffer_y)

    return frame_x1, frame_y1, frame_x2, frame_y2

def get_zoomed_frame(frame, width, height, x1, y1, x2, y2):
    # Define the zoomed frame
    zoomed_frame = frame[y1:y2, x1:x2]
    zoomed_frame = cv2.resize(zoomed_frame, (width, height))
    _, buffer = cv2.imencode('.jpeg', zoomed_frame)
    return buffer.tobytes()