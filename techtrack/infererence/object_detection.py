import os
import cv2
import numpy as np
import json
from copy import deepcopy

class Model:
    def __init__(self, model_config, model_weights, class_names):
        # Initiate a YOLO model
        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
        # Return names of all layers
        self.layer_names = self.net.getLayerNames()

        # Get the output layers and handle different formats
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        
        # If unconnected_out_layers is a 2D array, use i[0], otherwise just use i
        if isinstance(unconnected_out_layers, np.ndarray):
            self.output_layers = [self.layer_names[i - 1] for i in unconnected_out_layers.flatten()]
        else:
            self.output_layers = [self.layer_names[i - 1] for i in unconnected_out_layers]

        with open(class_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    
    def predict(self, preprocessed_frame):
        # Transform image - blob
        blob = cv2.dnn.blobFromImage(preprocessed_frame, 
                                     scalefactor=1/255, 
                                     size=(416, 416), 
                                     swapRB=True,
                                     crop=False)
        
        self.net.setInput(blob)

        # Forward pass to get predictions
        predictions = self.net.forward(self.output_layers)

        # Return a list of arrays where each array contains bounding box coordinates, \
        # objectness scores, and class probabilities for each detected object in the frame.
        return predictions
    

    def post_process(self, predict_output, objectiveness_threshold=0.5):
        bboxes = []
        class_ids = []
        scores = []

        for output in predict_output:
            for detection in output:
                prob_list = detection[5:] # Probabilities for classes

                if prob_list.size == 0:  # Check if the scores_list is empty
                    continue  # Skip this detection if no class probabilities are available

                class_id = np.argmax(prob_list) # take class id for highest probability
                objectiveness = detection[4]

                if objectiveness > objectiveness_threshold:
                    # Extract bounding box when threshold is met
                    center_x, center_y, box_width, box_height = detection[0:4]
                    bbox = [center_x, center_y, box_width, box_height]
                    
                    # # Check if class_id is already added, if yes, replace if confidence is higher
                    # if class_id in class_ids:
                    #     existing_index = class_ids.index(class_id)
                    #     if confidence > scores[existing_index]:  # Keep the higher confidence
                    #         bboxes[existing_index] = bbox
                    #         scores[existing_index] = confidence
                    # else:
                    bboxes.append(bbox)
                    class_ids.append(class_id)
                    scores.append(float(objectiveness))

        return bboxes, class_ids, scores


# Get predicted raw data before post_process()
def pred_raw_data(images, image_names, yolo_model, objective_threshold=0.3):
    pred_raw_data = {}

    for image, image_name in zip(images, image_names):
        # Process the image
        image_resized = cv2.resize(image, (416, 416))
        
        # Get predicted raw data
        predict_output = yolo_model.predict(image_resized)
        bboxes = []
        class_index = []
        objectness_scores = []
        class_probabilities = []

        for output in predict_output:
            for detection in output:
                bbox = detection[0:4]
                objectness = detection[4]
                prob_list = detection[5:]  # Class probabilities
                class_id = np.argmax(prob_list)
                class_prob = prob_list[class_id]

                # Only save prediction information for highly objectiveness score
                if objectness > objective_threshold:
                    bboxes.append(bbox)
                    objectness_scores.append(objectness)
                    class_index.append(class_id)
                    class_probabilities.append(class_prob)

        # Save predictions for this image
        pred_raw_data[image_name] = {
            'bboxes': np.array(bboxes),
            'objectness_scores': np.array(objectness_scores),
            'class_index': np.array(class_index),
            'class_prob': np.array(class_probabilities)
        }
    
    return pred_raw_data


# Save the predicted_raw data into text files
def save_predicted_raw(predicted_raw, output_dir, image_names):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each image's predictions and save them as text files
    for name, prediction_data in predicted_raw.items():
        if name in image_names:
            try:
                # Create a filepath to save the file
                output_file = os.path.join(output_dir, f"{name}.txt")
                
                # Open the file for writing
                with open(output_file, 'w') as f:
                    # For each predicted bounding box and its corresponding score and class probability
                    for bbox, objectness_scores, class_id, class_prob in \
                    zip(prediction_data['bboxes'], prediction_data['objectness_scores'], prediction_data['class_index'], prediction_data['class_prob']):
                        # Write class ID, normalized bounding box (x_center, y_center, width, height), and the score
                        bbox_str = " ".join(map(str, bbox))
                        f.write(f"{class_id} {bbox_str} {class_prob} {objectness_scores}\n")
            
            except FileNotFoundError as e:
                print(f"File not found error for {name}: {e}")
                continue  # Skip to the next image if file writing fails
    
    print(f"Predicted raw data saved to {output_dir}")


# Load predictions or annotations from a text file
def load_txt_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            data.append(values)
    return data


# Define IoU (Intersection over Union) for model comparison metric
def calculate_iou(box1, box2):
    # Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calculate the intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Calculate the union
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0
    iou = inter_area / union_area
    return iou


def match_box(box1, box2, iou_threshold=0.6):
    iou = calculate_iou(box1, box2)
    return True if iou > iou_threshold else False


def assign_true_object(predictions, annotations, iou_threshold=0.6):
    assigned_predictions = deepcopy(predictions)
    for object in assigned_predictions:
        pred_bbox = object[1:5]
        for true_obj in annotations:
            true_bbox = true_obj[1:5]
            # If predicted box can be matched to a true box, assign the true class and bbox
            if match_box(pred_bbox, true_bbox, iou_threshold):
                true_class = true_obj[0]
                object.append(true_class)
                object.append(true_bbox)
    # If associated, len(assigned_predictions)=9 
    # [pred_class, pred_x, pred_y, pred_w, pred_h, class_prob, objectiveness_score, true_class, true_bbox]
    return assigned_predictions


if __name__ == "__main__":
    # Assign to yolo_version to test 2 YOLO models
    yolo_version = 1

    yolo_model_folder = f'yolo_model_{yolo_version}'
    # Paths to the extracted files
    model_config = os.path.join(yolo_model_folder, f'yolov4-tiny-logistics_size_416_{yolo_version}.cfg')
    model_weights = os.path.join(yolo_model_folder, f'yolov4-tiny-logistics_size_416_{yolo_version}.weights')
    class_names = os.path.join(yolo_model_folder, 'logistics.names')

    yolo_model = Model(model_config, model_weights, class_names)

    frame_path = os.path.join('saved_frames', 'frame_2.jpg')
    frame = cv2.imread(frame_path)

    predictions = yolo_model.predict(frame)
    np.set_printoptions(threshold=np.inf)
    print(predictions)
    

    # bboxes, class_ids, scores = yolo_model.post_process(predictions, score_threshold=0.3)

    # # Output the filtered results
    # print("Bounding Boxes:", bboxes)
    # print("Class IDs:", class_ids)
    # print("Confidence Scores:", scores)
