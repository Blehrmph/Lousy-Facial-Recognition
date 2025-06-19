import face_recognition
import os


def treatedData(directory): # LOADS IMAGES AND ANALYZES THEM AND WHATNOT
    known_faces = []

    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        image = face_recognition.load_image_file(image_path)

        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:
            known_faces.append(face_encoding[0])
        else:
            print(f"{filename} is useless. Deleting...")
            delete = os.path.join(directory,filename)
            os.remove(delete)
    return known_faces