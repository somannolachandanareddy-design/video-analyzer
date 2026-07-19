pip install -q opencv-python pillow
import os
import cv2
from PIL import Image
import gradio as gr
from google import genai

# Gemini API Key
client = genai.Client(api_key="")

def analyze_video(video_path):

    if video_path is None:
        return "Please upload a video."

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prevent division by zero
    if fps == 0:
        fps = 30

    frame_interval = fps * 2   # One frame every 2 seconds

    frame_count = 0
    frame_number = 1

    descriptions = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:

            frame_path = f"frame_{frame_number}.jpg"
            cv2.imwrite(frame_path, frame)

            image = Image.open(frame_path)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    "Describe what is happening in this frame in detail.",
                    image
                ]
            )

            descriptions.append(
                f"Frame {frame_number}:\n{response.text}\n"
            )

            os.remove(frame_path)

            frame_number += 1

        frame_count += 1

    cap.release()

    if not descriptions:
        return "No frames were extracted."

    return "\n".join(descriptions)


demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(
        label="Scene Understanding",
        lines=20
    ),
    title="🎥 Video Scene Understanding with Gemini",
    description="Upload a video and Gemini will analyze one frame every 2 seconds."
)

demo.launch()
