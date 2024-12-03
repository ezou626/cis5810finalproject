import base64
import cv2
import google.generativeai as genai
gemini_model = genai.GenerativeModel(model_name = "gemini-1.5-flash")

def extract_frames(video_url):
    video_capture = cv2.VideoCapture(video_url)
    success, frame = video_capture.read()
    while success:
        yield frame
        success, frame = video_capture.read()
    video_capture.release()

def batch_frames(frames, batch_size, framerate):
    batch = []
    for i, frame in enumerate(frames):
        if i % framerate == 0:
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

async def caption_images_with_gemini(images, width: int, height: int):
    instruction = "Describe the sequence occuring the in these images in a single detailed sentence. Do not include any other words in your response besides the sentence."
    prompt = [
        {
            'mime_type':'image/jpeg', 
            'data': base64.b64encode(
                        cv2.imencode('.jpeg', 
                                     cv2.resize(image, (width, height))
                    )[1]).decode()
        } for image in images
    ]
    prompt.append(instruction)
    response = await gemini_model.generate_content_async(prompt)
    return response.text.strip()

async def get_caption_for_batch(images, width: int, height: int):
    """
    Generate caption for a batch of images

    Args:
        images (list[np.ndarray]): A list of images as NumPy arrays.

    Returns:
        list[str]: A list of generated captions.
    """
    return await caption_images_with_gemini(images, width, height)