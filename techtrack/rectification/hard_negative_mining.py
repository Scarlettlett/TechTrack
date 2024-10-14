'''This file contains method on loss calculation for hard negative mining module'''
import numpy as np
import cv2
import sys
import os
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from infererence.object_detection import *


# Compute the Mean Squared Error between the predicted and ground truth bboxes
def compute_mse(assigned_predictions):
    mse = 0
    for object in assigned_predictions:
        pred_bbox = np.array(object[1:5])
        # If associated, len(assigned_predictions)=9 
        # [pred_class, pred_x, pred_y, pred_w, pred_h, class_prob, objectiveness_score, true_class, true_bbox]
        if len(object) == 9:
            true_bbox = np.array(object[-1])
            mse += np.mean((pred_bbox - true_bbox) ** 2)
        # No associated true box
        else:
            mse += np.mean((pred_bbox) ** 2)
    
    return mse


# Calculate cross entropy loss for classification
def calculate_entropy_loss(assigned_predictions, epsilon=1e-10):
    entropy_loss = 0
    for object in assigned_predictions:
        # Predicted box and true box have overlap more than the iou threshold
        if len(object) == 9:
            # Check if predicted class agree to true class
            if object[0] == object[-2]:
                # Use class probabilities for cross entropy loss calculation
                entropy_loss += -np.log(object[5] + epsilon)
        
    return entropy_loss


# Compute the total loss (MSE for bounding box + Cross-Entropy for classification)
def compute_loss(predictions, annotations, iou_threshold=0.6):
    assigned_predictions = assign_true_object(predictions, annotations, iou_threshold=iou_threshold)

    class_loss = calculate_entropy_loss(assigned_predictions)
    mse = compute_mse(assigned_predictions)

    total_loss = class_loss + mse

    return total_loss


# Identify hard images based on the total loss for all objects in each image
def sample_hard_negatives(prediction_dir: str, annotation_dir: str, num_samples: int, iou_threshold=0.6, loss_threshold=1):
    image_losses = []
    hard_negative_count = 0
    total_images = 0
    cumulative_loss = 0

    # Iterate through each file in the prediction directory
    for pred_file in os.listdir(prediction_dir):
        if pred_file.endswith(".txt"):
            pred_path = os.path.join(prediction_dir, pred_file)
            annotation_path = os.path.join(annotation_dir, pred_file)

            if not os.path.exists(annotation_path):
                print(f"Annotation file not found for {pred_file}, skipping.")
                continue

            # Load predictions and annotations from the text files
            predictions = load_txt_file(pred_path)
            annotations = load_txt_file(annotation_path)

            # Compute total loss for this image
            total_loss = compute_loss(predictions, annotations, iou_threshold=iou_threshold)

            cumulative_loss += total_loss

            # Increment hard negative count if loss is above the threshold
            if total_loss > loss_threshold:
                hard_negative_count += 1

            # Store the file and its loss
            image_losses.append((pred_file, total_loss))
            total_images += 1

    # Calculate the average loss across all images
    average_loss = cumulative_loss / total_images if total_images > 0 else 0
    print(f"Average Loss: {average_loss:.4f}")

    # Calculate the hard negative error rate
    hard_negative_error_rate = hard_negative_count / total_images if total_images > 0 else 0
    print(f"Hard Negative Error Rate: {hard_negative_error_rate:.2%}")

    # Sort the images by total loss (descending order)
    image_losses.sort(key=lambda x: x[1], reverse=True)

    # Get the top-N hardest images
    hard_images = [x[0] for x in image_losses[:num_samples]]

    return hard_images

