import cv2
import math
import cvzone
import torch
from ultralytics import YOLO
from datetime import datetime
import os

# Class names for detection
classNames = ['chit', 'phone', 'hand', 'peeking', 'supplement-passing']

# Print current working directory to help with debugging
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Specify the path to your model file
# Option 1: Use absolute path (replace with your actual path)
model_path = r"/cheat_detection/ppedet\yolo11L without patience.pt"
# Option 2: Use relative path if the model is in the same directory as the script
# model_path = "retrained_yolo11L_100_epochs.pt"

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at: {model_path}")
    print("Please check if the path is correct and the file exists")
    exit()
else:
    print(f"Model file found at: {model_path}")

# Initialize webcam (0 is usually the default webcam)
cat = cv2.VideoCapture(0)

if not cat.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Set up device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
try:
    model = YOLO(model_path).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Set up output directory and video writer
output_dir = "../weebcamera_output"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"webcam_recording_{timestamp}.mp4")

# Get webcam properties
frame_width = int(cat.get(3))
frame_height = int(cat.get(4))
fps = 30  # Fixed FPS for webcam recording

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, img = cat.read()
    if not success:
        print("Failed to grab frame from webcam")
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            if cls < len(classNames):
                label = f'{classNames[cls]} {conf}'
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=2, thickness=1)

    out.write(img)

    cv2.imshow("Webcam Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cat.release()
out.release()
cv2.destroyAllWindows()

if device == "cuda":
    torch.cuda.empty_cache()
print(f"Recording saved as: {output_path}")