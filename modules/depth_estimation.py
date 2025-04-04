# modules/depth_estimation.py
import numpy as np

def compute_depth_for_detections(detections, depth_frame):
    # For each detection, compute the average depth in the bounding box
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Crop the depth region corresponding to the detection
        roi = depth_frame[y1:y2, x1:x2]
        if roi.size > 0:
            # Compute average depth (you might want to use median to reduce noise)
            avg_depth = float(np.median(roi))
        else:
            avg_depth = 0.0
        det["depth"] = avg_depth
    return detections
