import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from ensemble_boxes import weighted_boxes_fusion
import argparse

def augment_image(image, hsv_shift: int, blur_ksize:int, noise_std: int, contrast_alpha: int, contrast_beta: int):
    
    augments = []
    
    ### HSV Augmentation
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_img[:,:,0] = (hsv_img[:,:,0] + hsv_shift) % 180
    hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    augments.append(hsv_img)
    
    ### Blur Augmentation
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_img = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    augments.append(blur_img)
    
    ### Noise Augmentation
    noise = np.random.normal(0, noise_std, image.shape).astype(np.uint8)
    noisy_img = cv2.add(image, noise)
    augments.append(noisy_img)
    
    ### Contrast Augmentation
    contrast_img = cv2.convertScaleAbs(image, alpha=contrast_alpha, beta=contrast_beta)
    augments.append(contrast_img)

    return augments
    
def predict_image(model, image, num_passes):
    
    model.model.train()
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            prediction = model.predict(image)
            predictions.append(prediction)
    
    model.model.eval()
    return predictions
    
def aggregate_predictions(predictions, original_shape):
    
    boxes_list = []
    scores_list = []
    labels_list = []
    
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
    img = cv2.imread(image_path)
    h, w, _ = img.shape  # Get the height and width of the image

    # Draw final detections on the image
    for box, score, label in zip(final_boxes, final_scores, final_labels):
        # Convert from normalized (xywhn) to pixel values
        x1, y1, x2, y2 = box
        x1 = int(x1 * w)  # Convert to pixel coordinates
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Class {int(label)}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Create directory if it doesn't exist
    save_dir = "results/"
    os.makedirs(save_dir, exist_ok=True)

    # Save the image in the specified directory
    save_path = os.path.join(save_dir, "final_result.jpg")
    cv2.imwrite(save_path, img)

    print(f"Final detection image saved at: {save_path}")

def main(model_path: str, image_path: str, num_passes: int):
    
    ### Load the model
    model = YOLO(model_path)
    
    ### Load the image
    original_image = cv2.imread(image_path)
    original_shape = original_image.shape
    all_images = augment_image(original_image, hsv_shift=20, blur_ksize=5, noise_std=20, contrast_alpha=1.5, contrast_beta=0)
    all_images.insert(0, original_image)
    
    ### Predict on the image with bayesian dropout enabled
    predictions = predict_image(model, all_images, num_passes)
    
    final_boxes, final_scores, final_labels = aggregate_predictions(predictions, original_shape)
    # print(final_boxes)
    
    plot_results(image_path, final_boxes, final_scores, final_labels)
    
if __name__ == '__main__':
    
    ### Taking input arguments from the user
    parser = argparse.ArgumentParser(description="Train YOLOv8 with custom parameters")

    parser.add_argument("--model_path", type=str, default="", help="Path to the model `model.pt`")
    parser.add_argument("--img_path", type=str, default="", help="Path to the test image")
    parser.add_argument("--num_passes", type=int, default=10, help="Number of passes for Bayesian Dropout")

    args = parser.parse_args()
    
    main(model_path=args.model_path,
         image_path=args.img_path,
         num_passes=args.num_passes)
