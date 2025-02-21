import torch
from ultralytics import YOLO
import os
from torch.optim.lr_scheduler import CyclicLR
import torch.optim.swa_utils as swa_utils
import logging
import argparse

from torch.utils.data import DataLoader
from ultralytics.data import YOLODataset

logging.getLogger("ultralytics").setLevel(logging.WARNING)

def main(data_path: str, model_path: str, save_dir: str, epochs: int, base_epoch: int, batch_size: int, image_size: int):
    
    os.makedirs(save_dir, exist_ok=True) ## Create save directory if it does not exist
    model = YOLO(model_path) ## Load model
    
    ### Defining cyclical LR
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.01, 
                                momentum=0.9)
    scheduler = CyclicLR(optimizer, 
                         base_lr=0.0001, 
                         max_lr=0.01, 
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
    
    swa_model = swa_utils.AveragedModel(model.model)
    swa_scheduler = swa_utils.SWALR(optimizer, anneal_strategy="cos", anneal_epochs=5, swa_lr=0.01)
    
    ### Training the model using SWA
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
        
        ### Applying SWA update after base_epoch
        if epoch > base_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        ### Save the SWA model periodically
        if epoch % 6 == 0:
            swa_model_path = f"{save_dir}/swa_model_{epoch}.pt"
            torch.save(swa_model.module.state_dict(), swa_model_path)
            print(f"\nSWA model saved at {swa_model_path}\n")
    
    ### Save the final SWA model
    final_swa_path = f"{save_dir}/final_swa_model.pt"
    torch.save(swa_model.module.state_dict(), final_swa_path)
    print(f"\nFinal SWA model saved at {final_swa_path}\n")
    
    print("\n Training Completed!!!\n")
     
if __name__ == "__main__":
    ### Taking input arguments from the user
    parser = argparse.ArgumentParser(description="Train YOLOv8 with custom parameters")

    parser.add_argument("--data_path", type=str, default="coco128.yaml", help="Path to data.yaml")
    parser.add_argument("--model_path", type=str, default="yolov8l.pt", help="Pretrained Model")
    parser.add_argument("--save_dir", type=str, default="swa_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--base_epoch", type=int, default=10, help="Base epoch for SWA")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=416, help="Image size for training")

    args = parser.parse_args()
    
    main(data_path=args.data_path,
         model_path=args.model_path,
         save_dir=args.save_dir,
         epochs=args.epochs,
         base_epoch= args.base_epoch,
         batch_size=args.batch_size,
         image_size=args.image_size)