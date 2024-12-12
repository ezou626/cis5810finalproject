from ultralytics import YOLO
import google.generativeai as genai
import os

_yolo_model = None
_google_client = None

def load_google_client():
    genai.configure(api_key=os.environ.get('G_API_KEY'))
    global _google_client
    _google_client = genai.GenerativeModel(model_name="gemini-1.5-flash")

def load_yolo_model():
    global _yolo_model
    _yolo_model = YOLO("yolov5s.pt", verbose=False)

def setup_models():
    load_google_client()
    load_yolo_model()
    print('models loaded successfully!')

def get_yolo_model():
    return _yolo_model

def get_google_client():
    return _google_client