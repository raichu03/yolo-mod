import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ensemble_boxes import weighted_boxes_fusion
import argparse

def apply_augmentations(image, augment_fn=None):
    """Applies user-defined or default augmentations."""
    if augment_fn:
        return augment_fn(image)
    
    # Default Augmentations using Albumentations
    transform = A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
    ])
    
    augmented = transform(image=image)
    return [augmented["image"]]

def predict_image(model, images, num_passes, tta_fn=None):
    """Performs multiple forward passes with optional test-time augmentations."""
    model.model.train()  # Enable dropout layers
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            augmented_images = [tta_fn(img) if tta_fn else img for img in images]  # Apply TTA if defined
            batch_predictions = model.predict(augmented_images)
            predictions.append(batch_predictions)
    
    model.model.eval()
    return predictions

def aggregate_predictions(predictions, original_shape):
    """Combines predictions from multiple passes using weighted boxes fusion."""
    boxes_list, scores_list, labels_list = [], [], []
    
    for result_list in predictions:
        for result in result_list:
            boxes = result.boxes.xywhn.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()

            # Rescale boxes to original image size
            h, w = original_shape[:2]
            scaled_boxes = boxes.copy()
            scaled_boxes[:, 0] *= w
            scaled_boxes[:, 1] *= h
            scaled_boxes[:, 2] *= w
            scaled_boxes[:, 3] *= h
            scaled_boxes[:, 0] -= scaled_boxes[:, 2] / 2
            scaled_boxes[:, 1] -= scaled_boxes[:, 3] / 2
            scaled_boxes[:, 2] += scaled_boxes[:, 0]
            scaled_boxes[:, 3] += scaled_boxes[:, 1]
            scaled_boxes[:, 0] /= w
            scaled_boxes[:, 1] /= h
            scaled_boxes[:, 2] /= w
            scaled_boxes[:, 3] /= h

            boxes_list.append(scaled_boxes)
            scores_list.append(scores)
            labels_list.append(labels)
    
    final_boxes, final_scores, final_labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=0.5, skip_box_thr=0.001
    )
    
    return final_boxes, final_scores, final_labels

def plot_results(image_path, final_boxes, final_scores, final_labels):
    """Draws the final detection results on the image."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape  # Get the height and width of the image

    for box, score, label in zip(final_boxes, final_scores, final_labels):
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)  
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class {int(label)}: {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    save_dir = "results/"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "final_result.jpg")
    cv2.imwrite(save_path, img)
    print(f"Final detection image saved at: {save_path}")

def main(model_path: str, image_path: str, num_passes: int, augment_fn=None, tta_fn=None):
    """Main function for loading the model, applying augmentations, and running inference."""
    model = YOLO(model_path)
    
    original_image = cv2.imread(image_path)
    original_shape = original_image.shape
    
    augmented_images = apply_augmentations(original_image, augment_fn)
    
    # Include the original image as well
    all_images = [original_image] + augmented_images
    
    predictions = predict_image(model, all_images, num_passes, tta_fn)
    
    final_boxes, final_scores, final_labels = aggregate_predictions(predictions, original_shape)
    
    plot_results(image_path, final_boxes, final_scores, final_labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 Inference with Custom Augmentations & TTA")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the model `model.pt`")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the test image")
    parser.add_argument("--num_passes", type=int, default=10, help="Number of passes for Bayesian Dropout")

    args = parser.parse_args()

    # Defined augmentation function using Albumentations
    def custom_augmentations(image):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.5)
        ])
        augmented = transform(image=image)
        return [augmented["image"]]

    # Defined Test-Time Augmentation (TTA)
    def custom_tta(image):
        """Applies a single test-time augmentation."""
        transform = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.RandomGamma(p=0.5),
        ])
        augmented = transform(image=image)
        return augmented["image"]

    main(
        model_path=args.model_path,
        image_path=args.img_path,
        num_passes=args.num_passes,
        augment_fn=custom_augmentations,
        tta_fn=custom_tta
    )
