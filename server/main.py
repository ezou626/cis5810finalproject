import asyncio
from asyncio import new_event_loop
from uvicorn import Server, Config
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys, time
import os
from dotenv import load_dotenv
from commentary import caption_images_with_blip, extract_frames, batch_frames, get_caption_for_batch
from frame_extractor import get_video_metadata
import google.generativeai as genai

# Load environment variables
load_dotenv('.env', override=True)
genai.configure(api_key=os.environ.get('G_API_KEY'))
print(os.environ.get('MACHINE_TYPE', 'not'), file=sys.stderr)

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
