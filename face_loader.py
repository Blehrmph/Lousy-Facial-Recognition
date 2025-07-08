import face_recognition
import os
import pickle
import hashlib


def get_folder_hash(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            files.append(f"{filename}:{os.path.getmtime(file_path)}")
    
    folder_content = "|".join(sorted(files))
    return hashlib.md5(folder_content.encode()).hexdigest()


def load_cached_faces(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return None


def save_cached_faces(cache_file, data):
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass


def load_person_faces(person_folder, person_name):
    faces = []
    names = []
    
    for filename in os.listdir(person_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_path = os.path.join(person_folder, filename)
        
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            for encoding in encodings:
                faces.append(encoding)
                names.append(person_name)
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return faces, names


def load_all_faces(base_directory='C:\\Users\\Administrateur\\Desktop\\PEOPLE\\'):
    cache_file = os.path.join(base_directory, 'face_cache.pkl')
    
    if not os.path.exists(base_directory):
        print(f"Directory {base_directory} does not exist.")
        return [], []
    
    # Get all person folders
    person_folders = [f for f in os.listdir(base_directory) 
                     if os.path.isdir(os.path.join(base_directory, f))]
    
    if not person_folders:
        print("No person folders found.")
        return [], []
    
    # Check if we can use cached data
    cached_data = load_cached_faces(cache_file)
    current_hash = {}
    
    for person_name in person_folders:
        person_path = os.path.join(base_directory, person_name)
        current_hash[person_name] = get_folder_hash(person_path)
    
    # Use cache if nothing changed
    if cached_data and cached_data.get('hashes') == current_hash:
        print("Using cached face data...")
        return cached_data['faces'], cached_data['names']
    
    # Load fresh data
    print("Loading face data...")
    all_faces = []
    all_names = []
    
    for person_name in person_folders:
        person_path = os.path.join(base_directory, person_name)
        faces, names = load_person_faces(person_path, person_name)
        all_faces.extend(faces)
        all_names.extend(names)
        print(f"Loaded {len(faces)} faces for {person_name}")
    
    # Cache the results
    cache_data = {
        'faces': all_faces,
        'names': all_names,
        'hashes': current_hash
    }
    save_cached_faces(cache_file, cache_data)
    
    return all_faces, all_names