import os
import cv2
import face_recognition
import numpy as np
import serial
import time
import CaptureFaceData
from FaceRecognition import treatedData

ARDUINO_PORT = 'COM3'  # ARDUINO PORT
BAUD_RATE = 9600 # THAT THING IDK
arduino = None

def initialize_arduino():
    global arduino
    try:
        arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  
        print(f"Arduino connected on {ARDUINO_PORT}")
        return True
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        print("Face recognition will continue without Arduino control")
        return False

def send_servo_signal():
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.write(b'ROTATE\n') 
            print("Signal sent to Arduino - Servo rotating 90 degrees")
        except Exception as e:
            print(f"Error sending signal to Arduino: {e}")

def load_all_faces(base_directory):
    
    known_faces = []
    known_names = []
    
    if not os.path.exists(base_directory):
        print(f"Base directory {base_directory} does not exist.")
        return known_faces, known_names
    

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
    name = input("Enter individual's name: ")
    directory = CaptureFaceData.direct(name)
    
    if not os.path.exists(directory) or not os.listdir(directory):
        print(f"No images found for {name}. Starting image capture...")
        CaptureFaceData.captureData(directory, name)
    else:
        add_more = input(f"Images already exist for {name}. Add more images? (y/n): ").lower()
        if add_more == 'y':
            CaptureFaceData.captureData(directory, name)
    
    print(f"Face data for {name} is ready.")

def detect_faces():
    base_directory = 'C:\\Users\\Administrateur\\Desktop\\PEOPLE\\'
    
    print("Loading all known faces...")
    known_faces, known_names = load_all_faces(base_directory)
    
    if not known_faces:
        print("No face data found. Add some faces first.")
        return
    
    print(f"Loaded {len(known_faces)} face encodings for {len(set(known_names))} people")
    print(f"Known people: {list(set(known_names))}")
    
    arduino_connected = initialize_arduino()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting face detection. Press 'q' to quit.")
    
    last_detection_time = 0
    detection_cooldown = 3  #
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # COMPARE FACES
        known_face_detected = False
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            detected_name = "Unknown"

            if True in matches:  # CHECK FOR MATCHES
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    detected_name = known_names[best_match_index]
                    known_face_detected = True

            color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, detected_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        current_time = time.time()
        if known_face_detected and arduino_connected:
            if current_time - last_detection_time > detection_cooldown:
                send_servo_signal()
                last_detection_time = current_time

        cv2.imshow('Multi-Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if arduino and arduino.is_open:
        arduino.close()
        print("Arduino connection closed")
    
    print("Face detection stopped.")

def main():
    print("=== Face Recognition System with Arduino Control ===")
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
            print("Invalid choice. Enter 1 or 2.")

if __name__ == "__main__":
    main()