# modules/grasp_detection.py
def detect_grasp_point(frame, detection):
    # For simplicity, return the center point of the bounding box as the grasp point.
    x1, y1, x2, y2 = detection["bbox"]
    grasp_x = (x1 + x2) // 2
    grasp_y = (y1 + y2) // 2
    return (grasp_x, grasp_y)
