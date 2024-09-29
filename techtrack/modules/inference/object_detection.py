import os
import cv2
import numpy as np

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

    def post_process(self, predict_output, score_threshold):
        bboxes = []
        class_ids = []
        scores = []

        for output in predict_output:
            for detection in output:
                # print(f"Detection: {detection}")

                scores_list = detection[5:] # Probabilities for classes

                if scores_list.size == 0:  # Check if the scores_list is empty
                    continue  # Skip this detection if no class probabilities are available

                class_id = np.argmax(scores_list) # take class id for highest probability
                confidence = scores_list[class_id]

                if confidence > score_threshold:
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
                    bboxes.append([center_x, center_y, box_width, box_height])
                    class_ids.append(class_id)
                    scores.append(float(confidence))

        return bboxes, class_ids, scores


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
    # print(predictions)

    bboxes, class_ids, scores = yolo_model.post_process(predictions, score_threshold=0.3)

    # Output the filtered results
    print("Bounding Boxes:", bboxes)
    print("Class IDs:", class_ids)
    print("Confidence Scores:", scores)
