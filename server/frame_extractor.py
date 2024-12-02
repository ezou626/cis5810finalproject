import cv2
import os
import numpy as np
import asyncio
from dotenv import load_dotenv

load_dotenv('./server/.env', override=True)

#test_stream: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'

def get_video_metadata(video_url: str) -> tuple[int, int, int]:
    """Get video metadata from a given URL

    Args:
        video_url (str): a URL to a video stream

    Returns:
        tuple[int, int, int]: a tuple containing the framerate, width, and height of the video stream
    """
    cap = cv2.VideoCapture(video_url)
    framerate = round(cap.get(5))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return framerate, width, height

async def frame_generator(video_url: str):
    """Streams 1 video frame per second from a given URL in groups

    Args:
        video_url (str): a URL to a video stream

    Yields:
        np.ndarray: a numpy image in BGR format
    """
    _, width, height = get_video_metadata(video_url)

    # start ffmpeg
    command = [
        os.environ.get('FFMPEG_PATH'),
        '-max_delay', '30000000',   # 30 seconds
        '-i', video_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-an', 
        'pipe:'
    ]
    ffmpeg_process = await asyncio.create_subprocess_exec(
        *command, 
        stdout=asyncio.subprocess.PIPE
    )
    try:
        while True:
            raw_frame = await ffmpeg_process.stdout.read(width * height * 3)
            if len(raw_frame) != (width * height * 3):
                # End of stream or error occurred
                break
            yield np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        return
    except Exception as e:
        print(e)
    finally:
        ffmpeg_process.terminate()
        ffmpeg_process.wait()
        return