AI-Based Vehicle Counter for Traffic Management

A real-time vehicle detection and counting system built with YOLOv8, OpenCV, and the SORT tracking algorithm. This project is a module of a larger AI-based traffic management system, focused on accurately counting vehicles entering and exiting a monitored road zone.


Tech Stack:

ToolPurposeYOLOv8 (Ultralytics) Real-time object detection, OpenCV Video processing & visualization, SORT + Kalman Filter Multi-object tracking, NumPyArray operations & detection formatting cvzone Bounding box & text rendering.

What It Does:
Detects vehicles (cars, trucks, buses, motorbikes) in a video feed using YOLOv8
Tracks each vehicle across frames with a unique ID using the SORT algorithm
Counts vehicles crossing virtual entry and exit trip lines
Applies a region-of-interest mask to focus detection on the relevant road zone only
Displays a live count overlay on the video feed


About the Approach:
- YOLOv8 for Real-Time Object Detection
- ROI Masking with cv2.bitwise_and
- SORT Algorithm — Multi-Object Tracking
- Virtual Trip Lines for Zone-Based Counting
- Centroid-Based Crossing Detection
    pythoncx, cy = x1 + w // 2, y1 + h // 2
    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


How to run:
  
- Install dependencies:
      pip install ultralytics opencv-python cvzone numpy
- Change the path of Video and Mask
- Run the counter:
      python traffic.py


  Credits:
  sort.py was not written as part of this project.
  GitHub: https://github.com/abewley/sort
