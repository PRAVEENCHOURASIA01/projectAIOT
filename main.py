# main.py
import cv2
from modules import vision_system, object_detection, depth_estimation, object_tracking, grasp_detection, facial_recognition, voice_command

def main():
    # Initialize RealSense camera
    pipeline, align = vision_system.initialize_camera()
    
    # Initialize YOLOv8 model (assumes a pre-trained model is downloaded)
    model = object_detection.load_yolo_model("yolov8n.pt")
    
    # Initialize tracker (using OpenCV's built-in tracker for example)
    tracker = object_tracking.initialize_tracker()
    
    # Start voice command listener (runs in background or thread)
    voice_command.start_voice_listener()
    
    try:
        while True:
            # Get a new frame from RealSense
            frames = vision_system.get_frames(pipeline, align)
            color_frame = frames['color']
            depth_frame = frames['depth']
            
            # Object detection and recognition
            detections = object_detection.detect_objects(model, color_frame)
            color_frame = object_detection.draw_detections(color_frame, detections)
            
            # Depth estimation for each detected object
            detections = depth_estimation.compute_depth_for_detections(detections, depth_frame)
            
            # Object tracking (initialize or update based on detections)
            tracker, tracked_objects = object_tracking.track_objects(tracker, color_frame, detections)
            color_frame = object_tracking.draw_tracks(color_frame, tracked_objects)
            
            # Grasp detection (calculate grasp points)
            for det in detections:
                grasp_point = grasp_detection.detect_grasp_point(color_frame, det)
                cv2.circle(color_frame, grasp_point, 5, (0, 255, 0), -1)
            
            # Facial recognition and interaction
            faces = facial_recognition.detect_and_recognize_faces(color_frame)
            color_frame = facial_recognition.draw_faces(color_frame, faces)
            
            # Display the resulting frame
            cv2.imshow("Robot Vision", color_frame)
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vision_system.cleanup(pipeline)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
