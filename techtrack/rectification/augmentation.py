'''This file contains 6 functions on apply transformation to a image'''

import cv2
import numpy as np
import os

# Flip the image horizontally
def horizontal_flip(image):
    return cv2.flip(image, 1)

# Apply a Gaussian blur to smooth the image
def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Resize the image to the specified size
def resize(image, new_size):
    return cv2.resize(image, new_size)

# Crop the image to the specified size
def random_crop(image, crop_size):
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    start_y = np.random.randint(0, h - crop_h + 1)
    start_x = np.random.randint(0, w - crop_w + 1)
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]

# Rotate the image by a given angle
def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# Increase or decrease the brightness of the image
def adjust_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


if __name__ == "__main__":
    frame_path = os.path.abspath(os.path.join('..', 'techtrack', 'saved_frames', 'frame_2.jpg'))
    image = cv2.imread(frame_path)

    # Apply horizontal flip
    flipped_image = horizontal_flip(image)

    # Apply Gaussian blur
    blurred_image = gaussian_blur(image)

    # Resize the image
    resized_image = resize(image, (300, 300))

    # Random crop
    cropped_image = random_crop(image, (200, 200))

    # Rotate the image by 45 degrees
    rotated_image = rotate(image, 45)

    # Adjust brightness
    bright_image = adjust_brightness(image, 50)

    # Display the result for each transformation
    cv2.imshow('Flipped Image', flipped_image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.imshow('Resized Image', resized_image)
    cv2.imshow('Cropped Image', cropped_image)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.imshow('Bright Image', bright_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()