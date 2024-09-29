import os
import cv2
import zipfile
from PIL import Image
from io import BytesIO
import numpy as np

def capture_video(filename, drop_rate):
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Could not open video {filename}.")
    
    frame_count = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
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


def load_ground_truth(file):
    ground_truths = []

    for line in file.readlines():
        class_id, x, y, width, height = map(float, line.strip().split())
        ground_truths.append([x, y, width, height, class_id])

    return ground_truths


if __name__ == "__main__":

    video_file = os.path.join('test_videos', 'worker-zone-detection.mp4')
    # video_file = os.path.join('test_videos', 'Safety_Full_Hat_and_Vest.mp4')
    drop_rate = 500

    # # Check total number of frames in the video 
    # # (4548 for worker-zone-detection.mp4)
    # # (1950 for Safety_Full_Hat_and_Vest.mp4)
    # cap = cv2.VideoCapture(video_file)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total number of frames in the video: {total_frames}")

    save_dir = 'saved_frames'
    os.makedirs(save_dir, exist_ok=True)

    saved_frame_count = 0

    for frame in capture_video(video_file, drop_rate):
        cv2.imshow('Frame', frame)

        # save the fram in the folder
        frame_filename = os.path.join(save_dir, f'frame_{saved_frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

