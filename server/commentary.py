import numpy as np
import pyttsx3
import os, time, sys

engine = pyttsx3.init()

def caption_images(images: list[np.ndarray]) -> tuple[str, bytes | None]:
    # makes the network call to the captioning model, should return audio
    start = time.time()
    engine.save_to_file('Hello World', 'text.mp3')
    engine.runAndWait()
    os.remove('text.mp3')
    end = time.time()
    print(f"Time taken to generate audio: {end - start}", file=sys.stderr)
    return "This is a caption", None