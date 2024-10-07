import os
import cv2
import zipfile
from PIL import Image
from io import BytesIO
import numpy as np

def capture_video(filename, drop_rate):
    cap = cv2.VideoCapture(filename, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"Could not open video {filename}.")
    
    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Failed to retrieve frame. Ending capture.")
            break

        # Only yield frame at each drop_rate
        if frame_count % drop_rate == 0:
            yield frame

        frame_count += 1
    
    cap.release()


def load_images_from_zip(zip_file_path, num_images=None):
    images = []
    image_names = []

    # Open the zip file and process .txt files
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')][:num_images]

        for file_name in image_files:
            with zip_ref.open(file_name) as file:
                img_data = file.read()
                img = Image.open(BytesIO(img_data))
                img = img.convert("RGB")
                img_cv = np.array(img)
                # Convert RGB to BGR for OpenCV uses BGR format)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                images.append(img_cv)
                # Append the image name (without the .jpg extension)
                image_name, _ = os.path.splitext(file_name)
                image_names.append(image_name)

    return images, image_names

def load_image_and_name(folder_path):
    images = []
    image_names = []
    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file is an image
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            
            # Read the image using OpenCV
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
                image_names.append(os.path.splitext(file_name)[0])

    return images, image_names


# def load_ground_truth(file):
#     ground_truths = []

#     for line in file.readlines():
#         class_id, x, y, width, height = map(float, line.strip().split())
#         ground_truths.append([x, y, width, height, class_id])

#     return ground_truths

def load_ground_truth(folder_path, image_names):
    ground_truth = {}

    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file is a ground truth text file
        if file_name.endswith('.txt'):
            # Extract the base image name to use as the key
            name = os.path.splitext(file_name)[0]
            if name in image_names:
            
            # Build full path to the ground truth file
                ground_truth_path = os.path.join(folder_path, file_name)
            
            # Read the ground truth annotations from the file
                try:
                    with open(ground_truth_path, 'r') as file:
                        annotations = file.readlines()
                    # Store the annotations for this image
                    ground_truth[name] = [line.strip() for line in annotations]
                except FileNotFoundError:
                    print(f"File not found: {ground_truth_path}")

    return ground_truth


if __name__ == "__main__":

    video_file = 'udp://127.0.0.1:23000'
    drop_rate = 30
    save_dir = 'saved_frames'
    os.makedirs(save_dir, exist_ok=True)

    saved_frame_count = 0

    for frame in capture_video(video_file, drop_rate):
        # Display the frame
        cv2.imshow('Frame', frame)

        # save the fram in the folder
        frame_filename = os.path.join(save_dir, f'frame_{saved_frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Note: Start the UDP stream with ffmpeg in a terminal first:
# ffmpeg -re -i ./test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
# then run this script