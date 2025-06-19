import os
import cv2
import face_recognition
import numpy as np
import CaptureFaceData
from FaceRecognition import treatedData

name = input("Enter individual's name:")
directory = CaptureFaceData.direct(name)

if not os.listdir(directory):
    CaptureFaceData.captureData(directory,name)


known_faces = treatedData(directory)

cap = cv2.VideoCapture(0)

while True: # MAIN PROGRAM
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # COMPARE FACES
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        detected_name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            detected_name = name

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, detected_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)



    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()