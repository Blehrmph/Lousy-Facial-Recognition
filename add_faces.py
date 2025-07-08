import cv2
import os
import face_recognition
import numpy as np


def create_person_folder(name):
    directory = f'C:\\Users\\Administrateur\\Desktop\\PEOPLE\\'
    person_folder = os.path.join(directory, name)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder


def capture_face_data(folder_path, name): # CAPTURES FACE DATA FOR A PERSON
    
    count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])  
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Add Face Data - Press SPACE to capture, ESC to exit")
    
    print(f"Adding faces for {name}")
    print("Press SPACE to capture image, ESC to exit")
    
    captured_images = []  # STORES CAPTURED IMAGE PATHS
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # RPEVIEW
        cv2.imshow("Add Face Data - Press SPACE to capture, ESC to exit", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            img_name = f"{name}-{count}.jpg"
            save_path = os.path.join(folder_path, img_name)
            cv2.imwrite(save_path, frame)
            captured_images.append(save_path)
            print(f"Saved: {img_name}")
            count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    return captured_images


def get_reference_encoding(folder_path, name):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        image_path = os.path.join(folder_path, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                print(f"Using {filename} as reference image")
                return encodings[0]
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return None


def validate_and_clean_images(captured_images, folder_path, name): #VALIDATES CAPTURED IMAGES & REMOVES INVALID ONES
    if not captured_images:
        print("No images were captured.")
        return
    
    print(f"\nValidating {len(captured_images)} captured images...")
    
    reference_encoding = get_reference_encoding(folder_path, name) # GET REFERENCE ENCODING FROM EXISTING IMAGES
    
    if reference_encoding is None: # IF NO REFERENCE, USE FIRST VALID CAPTURED IMAGE AS REFERENCE
        for img_path in captured_images:
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    reference_encoding = encodings[0]
                    print(f"Using first valid captured image as reference")
                    break
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if reference_encoding is None:
            print("No valid face found in any captured image!")
            for img_path in captured_images: # DELETE ALL CAPTURED IMAGES
                try:
                    os.remove(img_path)
                except:
                    pass
            return
    
    valid_images = 0
    deleted_images = 0
    
    for img_path in captured_images:
        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                print(f"{os.path.basename(img_path)}: No face detected")
                os.remove(img_path)
                deleted_images += 1
                continue
            
            # CHECKS IF FACE MATCHES REFERENCE
            matches = face_recognition.compare_faces([reference_encoding], encodings[0], tolerance=0.6)
            
            if matches[0]:
                print(f"{os.path.basename(img_path)}: Valid face")
                valid_images += 1
            else:
                print(f"{os.path.basename(img_path)}: Different person detected")
                os.remove(img_path)
                deleted_images += 1
                
        except Exception as e:
            print(f"{os.path.basename(img_path)}: Error processing - {e}")
            try:
                os.remove(img_path)
                deleted_images += 1
            except:
                pass
    
    print(f"\nValidation complete:")
    print(f"Valid images: {valid_images}")
    print(f"Deleted images: {deleted_images}")
    
    if valid_images == 0:
        print("No valid images remain! Please try capturing again.")
    else:
        print(f"{name}'s face data is ready with {valid_images} valid images!")


def main(): # MAIN FUNCTION TO ADD FACE DATA
    
    name = input("\nEnter person's name: ").strip()
    if not name:
        print("Name cannot be empty")
        return
    
    folder_path = create_person_folder(name)
    
    # CHECKS IF PERSON ALREADY EXISTS
    existing_images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    if existing_images:
        print(f"Found {len(existing_images)} existing images for {name}")
        add_more = input("Add more images? (y/n): ").lower()
        if add_more != 'y':
            return
    
    # CAPTURE IMAGE FIRST
    captured_images = capture_face_data(folder_path, name)
    
    # VALIDATES AND CLEANS THEM
    validate_and_clean_images(captured_images, folder_path, name)


if __name__ == "__main__":
    main()