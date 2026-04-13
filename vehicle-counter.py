"""
AI-Based Vehicle Counter for Traffic Management
================================================
Counts vehicles entering and exiting a monitored zone using:
- YOLOv8 for real-time object detection
- SORT algorithm for multi-object tracking
- OpenCV for video processing and visualization
"""

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort


VIDEO_PATH = "assets/traffic.mp4"       
MASK_PATH  = "assets/mask.png"          
MODEL_PATH = "weights/yolov8l.pt"       

# Virtual trip lines [x1, y1, x2, y2]
ENTRY_LINE = [460, 200, 673, 200]
EXIT_LINE  = [100, 350, 677, 350]

# Detection confidence threshold
CONF_THRESHOLD = 0.3

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

CLASS_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def is_crossing_line(cx, cy, line, tolerance=15):
    """Check if a centroid crosses a horizontal virtual line."""
    x1, y1, x2, _ = line
    return x1 < cx < x2 and y1 - tolerance < cy < y1 + tolerance


def draw_ui(img, count):
    """Draw the vehicle count on the frame."""
    cv2.putText(
        img, f"Vehicles: {count}",
        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (0, 255, 255), 3
    )


def draw_lines(img, entry_color=(0, 0, 255), exit_color=(0, 0, 255)):
    """Draw entry and exit trip lines on the frame."""
    cv2.line(img, (ENTRY_LINE[0], ENTRY_LINE[1]), (ENTRY_LINE[2], ENTRY_LINE[3]), entry_color, 3)
    cv2.line(img,  (EXIT_LINE[0],  EXIT_LINE[1]),  (EXIT_LINE[2],  EXIT_LINE[3]),  exit_color, 3)


def main():
    cap   = cv2.VideoCapture(VIDEO_PATH)
    model = YOLO(MODEL_PATH)
    mask  = cv2.imread(MASK_PATH)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return
    if mask is None:
        print(f"[ERROR] Cannot load mask: {MASK_PATH}")
        return

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    counted_ids = []

    print("[INFO] Starting vehicle counter. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("[INFO] End of video stream.")
            break

        roi = cv2.bitwise_and(frame, mask)

        results    = model(roi, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = CLASS_NAMES[cls]

                if label in VEHICLE_CLASSES and conf > CONF_THRESHOLD:
                    detections = np.vstack(
                        (detections, np.array([x1, y1, x2, y2, conf]))
                    )

        tracked = tracker.update(detections)

        draw_lines(frame)

        for result in tracked:
            x1, y1, x2, y2, track_id = map(int, result)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(frame, f"ID {track_id}", (max(0, x1), max(35, y1)),
                               scale=1.5, thickness=2, offset=8)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if is_crossing_line(cx, cy, ENTRY_LINE):
                if track_id not in counted_ids:
                    counted_ids.append(track_id)
                    cv2.line(frame,
                             (ENTRY_LINE[0], ENTRY_LINE[1]),
                             (ENTRY_LINE[2], ENTRY_LINE[3]),
                             (0, 255, 0), 3)

            if is_crossing_line(cx, cy, EXIT_LINE):
                if track_id in counted_ids:
                    counted_ids.remove(track_id)
                    cv2.line(frame,
                             (EXIT_LINE[0], EXIT_LINE[1]),
                             (EXIT_LINE[2], EXIT_LINE[3]),
                             (0, 255, 0), 3)

        draw_ui(frame, len(counted_ids))
        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()