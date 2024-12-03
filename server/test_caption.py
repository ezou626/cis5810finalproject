import httpx
import os
import base64
import cv2

from dotenv import load_dotenv
load_dotenv('server/.env', override=True)

import google.generativeai as genai
genai.configure(api_key=os.environ.get('G_API_KEY'))

model = genai.GenerativeModel(model_name = "gemini-1.5-flash")

image = cv2.imread('C:/Users/ezou6/Documents/GitHub/cis5810finalproject/server/test_image.png')

prompt = "Caption this image."
response = model.generate_content([
    {
        'mime_type':'image/jpeg', 
        'data': base64.b64encode(cv2.imencode('.jpeg', image)[1]).decode()
    }, 
    prompt
])

print(response.text)