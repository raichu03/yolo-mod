import torch
from ultralytics import YOLO
import os
from torch.optim.lr_scheduler import CyclicLR
import torch.optim.swa_utils as swa_utils
import logging
import argparse


def main(data_path: str, model_path: str, save_dir: str, epochs: int, base_epoch: int, batch_size: int, image_size: int, 
         base_lr: float, max_lr: float, flipud: float, fliplr: float, mosaic: float, mixup: float, hsv_h: float, hsv_s: float, hsv_v: float):
    
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}\n")

    # Load the base model
    model = YOLO(model_path).to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=3, mode='triangular')

    # Train the base model
    model.train(data=data_path, 
                epochs=base_epoch, 
                batch=batch_size, 
                imgsz=image_size,
                flipud=flipud, 
                fliplr=fliplr, 
                mosaic=mosaic, 
                mixup=mixup, 
                hsv_h=hsv_h, 
                hsv_s=hsv_s, 
                hsv_v=hsv_v, 
                device=device, 
                save_period=10,
                project='train_logs',
                name=f"epoch_{base_epoch}" )

    # Save base model
    base_model_path = f"{save_dir}/base_model_{base_epoch}.pt"
    model.save(base_model_path)
    print(f"\nBase model saved at {base_model_path}\n")

    # Free memory
    del model
    torch.cuda.empty_cache()

    fge_model_paths = []

    # Train using FGE (each time, reload model from disk)
    for epoch in range(base_epoch + 1, epochs + 1):
        print(f"\nTraining Epoch: {epoch}")

        model = YOLO(base_model_path).to(device)  # Reload base model

        model.train(data=data_path, 
                    epochs=1, 
                    batch=batch_size, 
                    imgsz=image_size,
                    flipud=flipud, 
                    fliplr=fliplr, 
                    mosaic=mosaic, 
                    mixup=mixup, 
                    hsv_h=hsv_h, 
                    hsv_s=hsv_s, 
                    hsv_v=hsv_v,
                    device=device,
                    project='train_logs',
                    name=f"epoch_{epoch}")

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Save model checkpoint and free memory
        fge_model_path = f"{save_dir}/fge_model_{epoch}.pt"
        model.save(fge_model_path)
        fge_model_paths.append(fge_model_path)
        base_model_path = fge_model_path

        del model
        torch.cuda.empty_cache()

    print("\nFGE training completed! Now applying SWA...\n")

    # Load the first FGE model as the SWA base
    model = YOLO(fge_model_paths[0]).to(device)    
    swa_model = swa_utils.AveragedModel(model.model)
    swa_scheduler = swa_utils.SWALR(optimizer, anneal_strategy="cos", anneal_epochs=5, swa_lr=0.01)

    # Load and average FGE models
    for fge_path in fge_model_paths[1:]:
        fge_model = YOLO(fge_path).to(device)
        swa_model.update_parameters(fge_model)
        swa_scheduler.step()
        del fge_model
        torch.cuda.empty_cache()

    # Save final SWA model
    final_swa_path = f"{save_dir}/final_swa_model.pt"
    model.save(final_swa_path)
    
    print(f"\nFinal SWA model saved at {final_swa_path}\n")
    print("\n Training Completed!!!\n")
     
if __name__ == "__main__":
    ### Taking input arguments from the user
    parser = argparse.ArgumentParser(description="Train YOLOv8 with FGE and SWA")

    parser.add_argument("--data_path", type=str, default="coco128.yaml", help="Path to data.yaml")
    parser.add_argument("--model_path", type=str, default="yolov8l.pt", help="Pretrained Model")
    parser.add_argument("--save_dir", type=str, default="model_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--base_epoch", type=int, default=10, help="Base epoch for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=416, help="Image size for training")
    parser.add_argument("--base_lr", type=float, default=0.0001, help="Low learning rate for FGE")
    parser.add_argument("--max_lr", type=float, default=0.01, help="High learning rate for FGE")
    parser.add_argument("--flipud", type=float, default=0.5, help="Flip up-down probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left-right probability")
    parser.add_argument("--mosaic", type=float, default=0.5, help="Mosaic probability")
    parser.add_argument("--mixup", type=float, default=0.5, help="Mixup probability")
    parser.add_argument("--hsv_h", type=float, default=0.015, help="Hue saturation value")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="Hue saturation value")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="Hue saturation value")

    args = parser.parse_args()
    
    main(data_path=args.data_path,
         model_path=args.model_path,
         save_dir=args.save_dir,
         epochs=args.epochs,
         base_epoch= args.base_epoch,
         batch_size=args.batch_size,
         image_size=args.image_size,
         base_lr=args.base_lr,
         max_lr=args.max_lr,
         flipud=args.flipud,
         fliplr=args.fliplr,
         mosaic=args.mosaic,
         mixup=args.mixup,
         hsv_h=args.hsv_h,
         hsv_s=args.hsv_s,
         hsv_v=args.hsv_v
         )
