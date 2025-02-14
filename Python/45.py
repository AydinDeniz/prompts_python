
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import os

# Define Deepfake Detection Model
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# Load Pretrained Model
def load_model(model_path="deepfake_detector.pth"):
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Preprocess Image for Model
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

# Detect Deepfake in Video
def detect_deepfake(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fake_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output).item()

        if prediction == 1:
            fake_count += 1
            cv2.putText(frame, "FAKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Deepfake Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    fake_percentage = (fake_count / frame_count) * 100
    print(f"Deepfake Probability: {fake_percentage:.2f}%")

if __name__ == "__main__":
    model_path = "deepfake_detector.pth"  # Replace with actual trained model path
    video_file = "input_video.mp4"  # Replace with actual video path

    if not os.path.exists(model_path):
        print("Trained model not found! Please train the model before using it.")
    else:
        print("Loading Deepfake Detection Model...")
        model = load_model(model_path)

        print("Analyzing Video for Deepfakes...")
        detect_deepfake(video_file, model)
