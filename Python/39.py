
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import cv2
import numpy as np

# Connect to the drone
vehicle = connect("127.0.0.1:14550", wait_ready=True)

# Object detection setup
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Detect obstacles
def detect_obstacle(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:  # Adjust confidence threshold
                return True
    return False

# Adjust flight path
def adjust_flight():
    print("Obstacle detected! Adjusting flight path...")
    vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_relative_frame.lat + 0.0001,
                                               vehicle.location.global_relative_frame.lon, 20))

if __name__ == "__main__":
    print("Starting autonomous drone navigation...")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if detect_obstacle(frame):
            adjust_flight()
        
        time.sleep(1)  # Adjust frequency
    
    cap.release()
    vehicle.close()
