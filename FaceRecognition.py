import face_recognition
import os

def treatedData(directory): # LOADS IMAGES AND ANALYZES THEM AND WHATNOT
    known_faces = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return known_faces

    for filename in os.listdir(directory):
        # ONLY PROCESS IMAGE FILES
        if not filename.lower().endswith(valid_extensions):
            continue
            
        image_path = os.path.join(directory, filename)
        
        try:
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)
            
            if face_encoding:
                known_faces.append(face_encoding[0])
            else:
                print(f"{filename} has no detectable face. Consider deleting it.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return known_faces

def treatedDataWithNames(directory, person_name): 
    """Returns face encodings with corresponding person name"""
    known_faces = []
    known_names = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return known_faces, known_names

    for filename in os.listdir(directory):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        image_path = os.path.join(directory, filename)
        
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                for encoding in face_encodings:  # CHECKS FOR MULTIPLE FACES AT ONCE
                    known_faces.append(encoding)
                    known_names.append(person_name)
            else:
                print(f"{filename} has no detectable face.")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return known_faces, known_names

def getAllKnownFaces(base_directory):
    """Load all faces from all person directories"""
    all_faces = []
    all_names = []
    
    if not os.path.exists(base_directory):
        print(f"Base directory {base_directory} does not exist.")
        return all_faces, all_names
    
    for person_folder in os.listdir(base_directory):
        person_path = os.path.join(base_directory, person_folder)
        
        if os.path.isdir(person_path):
            print(f"Loading faces for {person_folder}...")
            faces, names = treatedDataWithNames(person_path, person_folder)
            all_faces.extend(faces)
            all_names.extend(names)
            print(f"Loaded {len(faces)} faces for {person_folder}")
    
    return all_faces, all_names