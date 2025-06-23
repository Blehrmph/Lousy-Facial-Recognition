import os
import cv2
import face_recognition
import numpy as np
import serial
import time
import CaptureFaceData
from FaceRecognition import treatedData

ARDUINO_PORT = 'COM9'  # ARDUINO PORT
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

def send_servo_signal(command):
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.write(f'{command}\n'.encode())  # SEND ARDUINO COMMAND
            if command == "KNOWN_FACE":
                print("Signal sent to Arduino - Moving servo to 90 degrees")
            elif command in ["NO_FACE", "UNKNOWN_FACE"]:
                print("Signal sent to Arduino - Moving servo to 0 degrees")
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
    
    # TRACKING PREVIOUS STATE
    previous_state = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # COMPARE FACES
        known_face_found = False
        unknown_face_found = False
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.6)
            detected_name = "Unknown"

            if True in matches:  # CHECK FOR MATCHES
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    detected_name = known_names[best_match_index]
                    known_face_found = True
                else:
                    unknown_face_found = True
            else:
                unknown_face_found = True

            color = (0, 255, 0) if detected_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, detected_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

     
        if arduino_connected:
            current_state = None
            
            if known_face_found:
                current_state = "KNOWN_FACE"
            elif unknown_face_found:
                current_state = "UNKNOWN_FACE"
            elif len(face_locations) == 0:
                current_state = "NO_FACE"
            

            if current_state != previous_state and current_state is not None:
                send_servo_signal(current_state)
                previous_state = current_state

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