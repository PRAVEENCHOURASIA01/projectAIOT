# modules/object_tracking.py
import cv2

def initialize_tracker():
    # Create a MultiTracker object
    multi_tracker = cv2.MultiTracker_create()
    return multi_tracker

def track_objects(multi_tracker, frame, detections):
    # If no trackers exist, initialize trackers based on current detections.
    if len(multi_tracker.getObjects()) == 0:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox = (x1, y1, x2 - x1, y2 - y1)
            tracker = cv2.TrackerCSRT_create()
            multi_tracker.add(tracker, frame, bbox)
    
    # Update tracker positions
    success, boxes = multi_tracker.update(frame)
    tracked_objects = []
    if success:
        for box in boxes:
            tracked_objects.append(box)
    return multi_tracker, tracked_objects

def draw_tracks(frame, tracked_objects):
    for box in tracked_objects:
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return frame
