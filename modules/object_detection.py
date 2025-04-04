# modules/object_detection.py
import cv2
from ultralytics import YOLO

def load_yolo_model(model_path: str):
    # Load pre-trained YOLOv8 model
    model = YOLO(model_path)
    return model

def detect_objects(model, frame):
    # Run inference
    results = model(frame)
    # results.xyxy[0] contains detections: [x1, y1, x2, y2, confidence, class]
    detections = []
    for result in results:
        for det in result.boxes.data.tolist():
            # Each det is [x1, y1, x2, y2, conf, cls]
            detections.append({
                "bbox": [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
                "confidence": det[4],
                "class": int(det[5])
            })
    return detections

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"ID:{det['class']} {det['confidence']:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame
