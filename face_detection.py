import cv2
import face_recognition
import numpy as np
from face_loader import load_all_faces


def detect_faces():
    print("Loading known faces...")
    known_faces, known_names = load_all_faces()
    
    if not known_faces:
        print("No face data found. Run add_faces.py first to add people.")
        return
    
    print(f"Loaded {len(known_faces)} face encodings for {len(set(known_names))} people")
    print(f"Known people: {list(set(known_names))}")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting face detection. Press 'q' to quit.")
    
    frame_skip = 2  # Process every 2nd frame
    frame_count = 0
    resize_factor = 0.5  # Resize frame for faster processing
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames for better performance
        if frame_count % frame_skip != 0:
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the small frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Scale back up face locations
        face_locations = [(int(top/resize_factor), int(right/resize_factor), 
                          int(bottom/resize_factor), int(left/resize_factor)) 
                         for top, right, bottom, left in face_locations]
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
            
            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")


if __name__ == "__main__":
    detect_faces()