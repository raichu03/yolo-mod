import os
import cv2
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import argparse

# Function to normalize bounding boxes for WBF
def normalize_boxes(boxes, img_width, img_height):
    return [[x/img_width, y/img_height, (x+w)/img_width, (y+h)/img_height] for x, y, w, h in boxes]

def main(dir_path: str, img_path: str):
    
    ### Load Models
    model_names = os.listdir(dir_path)
    fge_model_paths = [file for file in model_names if file.startswith("fge_")]
    model_paths = [os.path.join(dir_path, file) for file in fge_model_paths]
    models = [YOLO(path) for path in model_paths]
    
    ### Load Image
    img = cv2.imread(img_path)
    
    # Get original image dimensions
    orig_height, orig_width = img.shape[:2]
    
    # Collect predictions from all models
    all_boxes = []
    all_scores = [] 
    all_labels = []
    
    for model in models:
        results = model(img_path)[0]  # Get inference results
        pred_boxes = results.boxes.xyxy.cpu().numpy()  # Get (x1, y1, x2, y2) format boxes
        scores = results.boxes.conf.cpu().numpy()  # Confidence scores
        labels = results.boxes.cls.cpu().numpy()  # Class labels

        # Normalize bounding boxes to [0, 1] range
        norm_boxes = np.copy(pred_boxes)
        norm_boxes[:, [0, 2]] /= orig_width  # Normalize x1 and x2
        norm_boxes[:, [1, 3]] /= orig_height  # Normalize y1 and y2

        # Ensure values are strictly within [0, 1] to avoid WBF warnings
        norm_boxes = np.clip(norm_boxes, 0, 1)

        all_boxes.append(norm_boxes.tolist())
        all_scores.append(scores.tolist())
        all_labels.append(labels.tolist())
        

    # Aggregate predictions using Weighted Box Fusion (WBF)
    iou_thr = 0.5
    skip_box_thr = 0.001
    boxes, scores, labels = weighted_boxes_fusion(all_boxes, all_scores, all_labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    # Denormalize boxes back to original image scale
    boxes = np.array([[x1 * orig_width, y1 * orig_height, x2 * orig_width, y2 * orig_height] for x1, y1, x2, y2 in boxes])

    # Filter by confidence threshold
    conf_threshold = 0.5
    final_boxes, final_scores, final_labels = [], [], []
    for i, score in enumerate(scores):
        if score >= conf_threshold:
            final_boxes.append(boxes[i])
            final_scores.append(score)
            final_labels.append(labels[i])

    # Draw final detections on the image
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class {int(label)}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Create directory if it doesn't exist
    save_dir = "results/"
    os.makedirs(save_dir, exist_ok=True)

    # Save the image in the specified directory
    save_path = os.path.join(save_dir, "final_result.jpg")
    cv2.imwrite(save_path, img)

    print(f"Final detection image saved at: {save_path}")

if __name__ == '__main__':
    
    ### Taking input arguments from the user
    parser = argparse.ArgumentParser(description="Train YOLOv8 with custom parameters")

    parser.add_argument("--models_path", type=str, default="fge_checkpoints", help="Path to saved models")
    parser.add_argument("--img_path", type=str, default="human.jpg", help="Path to test image")

    args = parser.parse_args()
    
    main(dir_path=args.models_path, 
         img_path=args.img_path)