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


def load_model():
    global detection_model
    try:
        detection_model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        detection_model = None


def get_video_metadata(video_url: str) -> tuple[int, int, int]:
    """Get video metadata from a given URL."""
    cap = cv2.VideoCapture(video_url)
    framerate = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return framerate, width, height


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


