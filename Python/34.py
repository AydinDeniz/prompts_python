
import cv2
import whisper
import numpy as np
import os

# Load Whisper model
model = whisper.load_model("base")

# Extract frames from video
def extract_key_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1

    cap.release()
    return frames

# Generate transcript from audio
def transcribe_audio(video_path):
    audio_path = "audio.wav"
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y")
    result = model.transcribe(audio_path)
    os.remove(audio_path)
    return result["text"]

# Save key frames to disk
def save_key_frames(frames, output_folder="key_frames"):
    os.makedirs(output_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(f"{output_folder}/frame_{i}.jpg", frame)

if __name__ == "__main__":
    video_file = "input_video.mp4"
    
    print("Extracting key frames...")
    key_frames = extract_key_frames(video_file)
    save_key_frames(key_frames)

    print("Generating transcript...")
    transcript = transcribe_audio(video_file)

    with open("video_summary.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    print("Video summary saved to video_summary.txt")
