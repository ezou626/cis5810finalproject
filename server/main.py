from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/video')
async def get_video(url: str):

    async def iterfile():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
                    
    return StreamingResponse(iterfile(), media_type="video/mp4")