import cv2
import os
import numpy as np
try:
    from .object_detection import Model
except ImportError:
    from object_detection import Model

def filter(bboxes, class_ids, scores, nms_iou_threshold):
    filtered_bboxes = []
    filtered_class_ids = []
    filtered_scores = []

    boxes_cv2 = []

    for bbox in bboxes:
        x = bbox[0] - bbox[2] / 2 # Top-left x
        y = bbox[1] - bbox[3] / 2  # Top-left y
        width = bbox[2]
        height = bbox[3]
        boxes_cv2.append([x, y, width, height])

    # boxes_cv2 = np.array(boxes_cv2)
    indices = cv2.dnn.NMSBoxes(boxes_cv2, scores, score_threshold=0.0, nms_threshold=nms_iou_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            filtered_bboxes.append(bboxes[i])
            filtered_class_ids.append(class_ids[i])
            filtered_scores.append(scores[i])

    return filtered_bboxes, filtered_class_ids, filtered_scores


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

    bboxes, class_ids, scores = yolo_model.post_process(predictions, score_threshold=0.3)
    print('Before filter:')
    print(bboxes, class_ids, scores)

    filtered_bboxes, filtered_class_ids, filtered_scores = filter(bboxes, class_ids, scores, nms_iou_threshold=0.8)
    print('\nAefore filter:')
    print(filtered_bboxes, filtered_class_ids, filtered_scores)
