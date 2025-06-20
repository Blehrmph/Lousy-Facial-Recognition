import os
import cv2
import face_recognition
import numpy as np
import CaptureFaceData
from FaceRecognition import treatedData

def load_all_faces(base_directory):
    """Load all faces from all person folders"""
    known_faces = []
    known_names = []
    
    if not os.path.exists(base_directory):
        print(f"Base directory {base_directory} does not exist.")
        return known_faces, known_names
    
    # Get all person folders
    person_folders = [f for f in os.listdir(base_directory) 
                     if os.path.isdir(os.path.join(base_directory, f))]
    
    if not person_folders:
        print("No person folders found.")
        return known_faces, known_names
    
    print(f"Found {len(person_folders)} person folders: {person_folders}")
    
    for person_name in person_folders:
        person_directory = os.path.join(base_directory, person_name)
        print(f"Loading faces for {person_name}...")
        
        person_faces = treatedData(person_directory)
        
        for face_encoding in person_faces:
            known_faces.append(face_encoding)
            known_names.append(person_name)
        
        print(f"Loaded {len(person_faces)} faces for {person_name}")
    
    return known_faces, known_names

def add_new_person():
    """Add a new person's face data"""
    name = input("Enter individual's name: ")
    directory = CaptureFaceData.direct(name)
    
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"No images found for {name}. Starting image capture...")
        CaptureFaceData.captureData(directory, name)
    else:
        add_more = input(f"Images already exist for {name}. Add more images? (y/n): ").lower()
        if add_more == 'y':
            CaptureFaceData.captureData(directory, name)
    
    print(f"Face data for {name} is ready!")

def detect_faces():
    """Detect all known faces from the database"""
    base_directory = 'C:\\Users\\Administrateur\\Desktop\\PEOPLE\\'
    
    print("Loading all known faces...")
    known_faces, known_names = load_all_faces(base_directory)
    
    if not known_faces:
        print("No face data found. Please add some faces first.")
        return
    
    print(f"Loaded {len(known_faces)} face encodings for {len(set(known_names))} people")
    print(f"Known people: {list(set(known_names))}")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting face detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # COMPARE FACES
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            detected_name = "Unknown"

            if True in matches:  # CHECK FOR MATCHES
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    detected_name = known_names[best_match_index]

            color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, detected_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Multi-Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")

def main():
    """Main program with mode selection"""
    print("=== Face Recognition System ===")
    print("1. Add new faces")
    print("2. Detect faces")
    
    while True:
        choice = input("Select option (1 or 2): ").strip()
        
        if choice == '1':
            add_new_person()
            break
        elif choice == '2':
            detect_faces()
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()