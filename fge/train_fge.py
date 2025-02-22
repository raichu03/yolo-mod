import torch
from ultralytics import YOLO
import os
from torch.optim.lr_scheduler import CyclicLR
import logging
import argparse

logging.getLogger("ultralytics").setLevel(logging.WARNING)

def main(data_path: str, model_path: str, save_dir: str, epochs: int, base_epoch: int, batch_size: int, image_size: int, base_lr: float=0.0001, max_lr: float=0.01):
    
    os.makedirs(save_dir, exist_ok=True) ## Create save directory if it does not exist
    model = YOLO(model_path) ## Load model
    
    ### Defining cyclical LR
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.01, 
                                momentum=0.9)
    scheduler = CyclicLR(optimizer, 
                         base_lr=base_lr, 
                         max_lr=max_lr, 
                         step_size_up=3, 
                         mode='triangular')
    
    ### Selecting the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Model on: {device}\n")
    
    ### Training the model for initial epochs
    model.train(data=data_path, 
                epochs=base_epoch, 
                batch=batch_size, 
                imgsz=image_size, 
                device=device, 
                save_period=10,
                verbose=False)

    ### Saving the base model
    model.save(f"{save_dir}/base_model_{base_epoch}.pt")
    print(f"\nBase model saved at {save_dir}/base_model_{base_epoch}.pt\n")
    
    ### Training the model using FGE
    for epoch in range(base_epoch+1, epochs+1):
        print(f"\n\nTraining Epoch: {epoch}")
        model.train(data=data_path, 
                    epochs=1, 
                    batch=batch_size, 
                    imgsz=image_size, 
                    device=device,
                    verbose=False)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        ### Saving the FGE model
        if epoch % 6 == 0:
            model.save(f"{save_dir}/fge_model_{epoch}.pt")
            print(f"\nFGE model saved at {save_dir}/fge_model_{epoch}.pt\n")
    
    print("\n Training Completed!!!\n")
     
if __name__ == "__main__":
    ### Taking input arguments from the user
    parser = argparse.ArgumentParser(description="Train YOLOv8 with custom parameters")

    parser.add_argument("--data_path", type=str, default="coco128.yaml", help="Path to data.yaml")
    parser.add_argument("--model_path", type=str, default="yolov8l.pt", help="Pretrained Model")
    parser.add_argument("--save_dir", type=str, default="fge_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--base_epoch", type=int, default=10, help="Base epoch for FGE")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=416, help="Image size for training")
    parser.add_argument("--base_lr", type=float, default=0.0001, help="low learning rate for FGE")
    parser.add_argument("--max_lr", type=float, default=0.01, help="high learning rate for FGE")

    args = parser.parse_args()
    
    main(data_path=args.data_path,
         model_path=args.model_path,
         save_dir=args.save_dir,
         epochs=args.epochs,
         base_epoch= args.base_epoch,
         batch_size=args.batch_size,
         image_size=args.image_size,
         base_lr=args.base_lr,
         max_lr=args.max_lr)