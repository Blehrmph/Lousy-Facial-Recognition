import cv2
import os


def direct(destination_folder): # DIRECTORY
    directory = f'C:\\Users\\Administrateur\\Desktop\\PEOPLE\\'
    final_dir = os.path.join(directory,destination_folder)

    os.makedirs(final_dir, exist_ok=True)  # IN CASE FOLDER DOESN'T EXIST
    return final_dir



def captureData(directory,name): # TAKES PICS ON THE SPOT
    count = 0
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face")


    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Capture Face", frame)
        k = cv2.waitKey(1)

        if k % 256 == 27:  # ESC to exit
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:  # SPACE to capture
            img_name = f"{name}-{count}.jpg"
            save_path = directory + "\\" + img_name

            cv2.imwrite(save_path, frame)
            print(f"{img_name} saved.")
            count += 1
    cam.release()
    cv2.destroyAllWindows()


