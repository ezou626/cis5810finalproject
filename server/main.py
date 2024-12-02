import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from frame_extractor import frame_generator, get_video_metadata
from commentary import caption_images

import httpx
import sys, os, time
from asyncio import new_event_loop
from uvicorn import Config, Server
from dotenv import load_dotenv

load_dotenv('./.env', override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/play_video')
async def play_video(url: str):

    async def iterfile():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(iterfile(), media_type="video/mp4")

ANALYSIS_WINDOW = 2 # seconds
CAPTION_GRANULARITY = 1

@app.get('/captions')
async def get_captions(video_url: str):

    framerate, _, _ = get_video_metadata(video_url)

    async def caption_stream():
        raw_images = []
        image_index = 1
        times_sent = 0
        start = time.time()
        try:
            async for image in frame_generator(video_url):
                if image_index % (framerate * CAPTION_GRANULARITY) == 0:
                    raw_images.append(image)
                    image_index = 1
                image_index += 1
                if len(raw_images) == ANALYSIS_WINDOW:
                    # caption the images
                    img_end = time.time()
                    print(f"Time taken to get images: {img_end - start}", file=sys.stderr)
                    text, _ = await caption_images(raw_images)
                    raw_images = []
                    times_sent += 1
                    end = time.time()
                    print(f"Time taken to complete endpoint: {end - start}", file=sys.stderr)
                    start = time.time()
                    yield f"event: message\ndata: {text} {times_sent}\n\n"
        except Exception as e:
            print(e, file=sys.stderr)
            yield f"event: message\ndata: {e}\n\n"
                    
    return StreamingResponse(caption_stream(), media_type="text/event-stream")

@app.get('/captions_debug')
async def get_captions():

    index = 0
    async def caption_stream():
        nonlocal index  # Use nonlocal to modify index within the enclosing scope
        while True:  # Infinite loop to keep streaming data
            await asyncio.sleep(1)  # Simulate a delay before sending the next data
            index += 1  # Increment the index
            # Send the event in proper SSE format
            yield f"event: message\ndata: {index}\n\n"
                    
    return StreamingResponse(caption_stream(), media_type="text/event-stream")

# loop = new_event_loop()
# config = Config(app=app)
# server = Server(config=config)
# loop.run_until_complete(server.serve())