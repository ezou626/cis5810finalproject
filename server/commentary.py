import base64
import asyncio
import sys
import sys
import cv2
import google.generativeai as genai

def extract_frames(video_url):
    video_capture = cv2.VideoCapture(video_url)
    success, frame = video_capture.read()
    while success:
        yield frame
        success, frame = video_capture.read()
    video_capture.release()

async def batch_frames(frames, batch_size, framerate):
    batch = []
    i = 0
    for frame in frames:
        i += 1
        if i % framerate:
            continue
        i = 0
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
        await asyncio.sleep(1)
    if batch:
        yield batch

def construct_prompt(images, captions_list, width, height):
    prompt = [
        {
            'mime_type':'image/jpeg', 
            'data': base64.b64encode(
                        cv2.imencode('.jpeg', 
                                     cv2.resize(image, (width, height))
                    )[1]).decode()
        } for image in images
    ]
    if captions_list:
        prompt.append("\nYour job is to create commentary for the images provided as if you were Stephen A. Smith. This is your previous commentary on the sport being played in these images.\n")
        for caption in captions_list:
            prompt.append(f'"{caption}\n"')
    instruction = "Continue the commentary by describing the relevant sports action in the images in a single detailed sentence. Make sure to identify the stage of the sport based on the actions of the players (e.g. warmup, final seconds, etc.) Do not include any other words in your response besides the sentence, and speak as if you said all of the comments together in conversation. Also, do not start every sentence the same way, i.e. 'listen up, folks'. Make sure they flow logically together and sound natural when combined."
    prompt.append(instruction)
    return prompt

async def caption_images_with_gemini(images, width: int, height: int, captions_list: list[str], gemini_model: genai.GenerativeModel):
    prompt = construct_prompt(images, captions_list, width, height)
    print('request for caption made.', file=sys.stderr)
    response = await gemini_model.generate_content_async(prompt)
    print('\n CAPTION RESPONSE:', response.text, file=sys.stderr)
    return response.text.replace('\n', ' ').replace('\r', '')

def caption_images_with_gemini_sync(images, width: int, height: int, captions_list: list[str], gemini_model: genai.GenerativeModel):
    prompt = construct_prompt(images, captions_list, width, height)
    print('request for caption made.\n', file=sys.stderr)
    response = gemini_model.generate_content(prompt)
    print('\n CAPTION RESPONSE:\n', response.text, file=sys.stderr)
    return response.text.replace('\n', ' ').replace('\r', '')