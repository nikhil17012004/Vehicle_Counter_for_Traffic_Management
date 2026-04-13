AI-Based Vehicle Counter for Traffic Management

A real-time vehicle detection and counting system built with YOLOv8, OpenCV, and the SORT tracking algorithm. This project is a module of a larger AI-based traffic management system, focused on accurately counting vehicles entering and exiting a monitored road zone.

What It Does:
Detects vehicles (cars, trucks, buses, motorbikes) in a video feed using YOLOv8
Tracks each vehicle across frames with a unique ID using the SORT algorithm
Counts vehicles crossing virtual entry and exit trip lines
Applies a region-of-interest mask to focus detection on the relevant road zone only
Displays a live count overlay on the video feed


Tech Stack
ToolPurposeYOLOv8 (Ultralytics)Real-time object detectionOpenCVVideo processing & visualizationSORT + Kalman FilterMulti-object trackingNumPyArray operations & detection formattingcvzoneBounding box & text rendering
