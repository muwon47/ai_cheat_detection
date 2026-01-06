import cv2
import math
import cvzone
import torch
import os
import glob
from ultralytics import YOLO
from datetime import datetime


classNames = ['chit', 'phone', 'hand', 'peeking', 'supplement-passing']
input_video_path = "../vid/Videos/VID_20250204_115355.mp4"
cat = cv2.VideoCapture(input_video_path)

if not cat.isOpened():
    print("Error: Could not open video.")
    exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO("ajinkya_yolol_bsz6_70epochs.pt").to(device)
output_dir = "../finaloutput"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"processed_video_{timestamp}.mp4")

frame_width = int(cat.get(3))
frame_height = int(cat.get(4))
fps = int(cat.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    success, img = cat.read()
    if not success:
        break

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf < 0.35:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 255, 255), colorC=(255, 255, 255), rt=1, t=1)

            cls = int(box.cls[0])
            if cls < len(classNames):
                label = f'{classNames[cls]} {conf}'
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1.3, thickness=1,
                                   colorR=(139, 0, 0), colorB=(139, 0, 0), offset=3, border=1,
                                   colorT=(255, 255, 255))

    out.write(img)

    cv2.imshow("Processed Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cat.release()
out.release()
cv2.destroyAllWindows()

if device == "cuda":
    torch.cuda.empty_cache()
print(f"Processed video saved as: {output_path}")
