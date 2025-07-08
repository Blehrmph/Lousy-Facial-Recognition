import cv2
import face_recognition
import numpy as np
from face_loader import load_all_faces
import threading
import queue
import time

# VERY PROUD OF THIS MAIN MUAHAHAHAHAHA
#Something is still wrong tho

class FaceRecognitionThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, known_faces, known_names):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.known_faces = known_faces
        self.known_names = known_names
        self.running = True
        self.daemon = True
        
    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # PROCESS FRAME FOR FACE RECON
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=0)
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)
                    
                    results = []
                    for (top, right, bottom, left), face_encoding in zip(face_locations[:2], face_encodings[:2]):
                        # SCALE BACK UP
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.5)
                        name = "Unknown"
                        
                        if True in matches:
                            face_distances = face_recognition.face_distance(self.known_faces, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                                name = self.known_names[best_match_index]
                        
                        results.append((top, right, bottom, left, name))
                    
                    self.result_queue.put(results)
                else:
                    self.result_queue.put([])
                    
            except queue.Empty:
                continue
    
    def stop(self):
        self.running = False


def main(): # SUPA FAST FACE RECON
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
    
    # SETTING CAM PROPERTIES
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Starting ultra-fast face detection. Press 'q' to quit.")
    
    # CREATING QUEUES AND THREADS
    frame_queue = queue.Queue(maxsize=2)
    result_queue = queue.Queue()
    
    recognition_thread = FaceRecognitionThread(frame_queue, result_queue, known_faces, known_names)
    recognition_thread.start()
    
    last_results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # SEND EVERY FRAME FOR PROCESSING
        if not frame_queue.full():
            frame_queue.put(frame.copy())
        
        # GET RESULT IF AVAILABLE
        try:
            last_results = result_queue.get_nowait()
        except queue.Empty:
            pass
        
        for top, right, bottom, left, name in last_results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        cv2.imshow('Ultra-Fast Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    recognition_thread.stop()
    recognition_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")


if __name__ == "__main__":
    main()