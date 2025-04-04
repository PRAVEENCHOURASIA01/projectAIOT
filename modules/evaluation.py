# modules/evaluation.py
import cv2
import numpy as np
from modules import object_detection, depth_estimation, facial_recognition

def test_object_detection(model, sample_image):
    detections = object_detection.detect_objects(model, sample_image)
    assert isinstance(detections, list)
    print("Object Detection Test Passed")
    return detections

def test_depth_estimation(detections, sample_depth):
    detections = depth_estimation.compute_depth_for_detections(detections, sample_depth)
    for det in detections:
        assert "depth" in det
    print("Depth Estimation Test Passed")
    return detections

def test_facial_recognition(sample_image):
    faces = facial_recognition.detect_and_recognize_faces(sample_image)
    print("Facial Recognition Test Passed")
    return faces

if __name__ == "__main__":
    # Create dummy sample images for testing
    sample_image = cv2.imread("sample.jpg") if False else np.zeros((480, 640, 3), dtype=np.uint8)
    sample_depth = np.zeros((480, 640), dtype=np.uint16)
    
    # Load a dummy model (adjust the model path accordingly)
    model = object_detection.load_yolo_model("yolov8n.pt")
    
    detections = test_object_detection(model, sample_image)
    detections = test_depth_estimation(detections, sample_depth)
    faces = test_facial_recognition(sample_image)
