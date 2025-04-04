# modules/facial_recognition.py
import cv2
import face_recognition

# For facial recognition, you can preload known face encodings and names.
# This is a stub for demonstration.
known_face_encodings = []  # Load or compute these encodings
known_face_names = []      # Corresponding names

def detect_and_recognize_faces(frame):
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    faces = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        # Compare against known faces (if any)
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
        faces.append({
            "bbox": (left, top, right, bottom),
            "name": name
        })
    return faces

def draw_faces(frame, faces):
    for face in faces:
        left, top, right, bottom = face["bbox"]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, face["name"], (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
