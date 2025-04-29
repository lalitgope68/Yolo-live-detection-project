import cv2
import time
import os
from ultralytics import YOLO
import random

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Create output folder if not exists
if not os.path.exists('detections'):
    os.makedirs('detections')

# Webcam setup
cap = cv2.VideoCapture(1)

# Variables for FPS calculation
prev_time = 0

# Random colors for each class
colors = {}
for class_id in range(len(model.names)):
    colors[class_id] = [random.randint(0, 255) for _ in range(3)]

while True:
    success, frame = cap.read()
    if not success:
        break

    # Predict using YOLO model
    results = model(frame, stream=True)

    # Process detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            color = colors.get(cls, (255, 0, 255))

            # Draw Rectangle and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Save screenshot if person detected
            if label == "person" and confidence > 0.5:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f'detections/person_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"[INFO] Person detected, image saved: {filename}")

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('YOLOv8 Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
