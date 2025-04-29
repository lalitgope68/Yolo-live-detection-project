# YOLOv8 Live Detection Project

This project runs real-time object detection using the latest YOLOv8 model on your webcam.

## Setup Instructions

1. Install the required libraries:
   ```
   pip install ultralytics opencv-python
   ```

2. Run the script:
   ```
   python yolo_live_detection_pro.py
   ```

3. The detected 'person' images will be saved automatically inside the `detections/` folder.

## Features
- Real-time detection with YOLOv8.
- Different color bounding boxes for each object.
- FPS display.
- Automatic screenshot saving for person detection.
