# yolo-ensemble modification
ensambling yolo model

### Installation
From the home directory of the project, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```
To train the model on your custom dataset, prepare the dataset that is suiitable for the yolo format. The dataset should be in the following format:
```bash
/dataset
  ├── images
  │   ├── train
  │   ├── val
  │   ├── test  (optional)
  ├── labels
  │   ├── train
  │   ├── val
  │   ├── test  (optional)
  ├── dataset.yaml
```
* The `images/` directory contains all images divided into `train`,`val/`, and optionally `test/` directories.
* The `labels/` directory contains annotation files in YOLO format.
* The `dataset.yaml` file defines class names and paths.

Once you prepare the dataset for training, you can train the YOLO model using various ensembling methods. 

## Fast Geometric Ensembling (FGE)

If you want to train the model using the FGE method, you can use the following command. Run this command from the **fge** direcotry:

```bash
python train_fge.py --data_path your_dataset.yaml --model_path your_custom_model --save_dir checkpoint_directory --epochs 300 --base_epoch 258 --batch_size 16 --image_size 416 --base_lr 0.0001 --max_lr 0.01
```
The arguments to pass duiing training are:
* `--data_path` : Path to the dataset.yaml file (**dataset.yaml**). Add your own directory or the code defaults to **coco128.yaml**.
* `--model_path` : Path to the custom model. You can add your custom model. The code defaults to **yolov8l.pt**.
* `--save_dir` : Directory to save the checkpoints. Add your own directory or the code defaults to **fge_checkpoints**.
* `--epochs` : Number of epochs to train the model. The code defaults to **300**.
* `--base_epoch` : The epoch number to start the ensembling and save the base model. The code defaults to **258**.
* `--batch_size` : Batch size for training. The code defaults to **16**.
* `--image_size` : Image size for training. The code defaults to **416**.
* `--base_lr` : The lower learning rate for training. The code defaults to **0.0001**.
* `--max_lr` : The base higher learning rate for training. The code defaults to **0.01**.

The model checkpoints are saved in the directory passed in the argument.

### Testing the FGE model on the test dataset
To test the FGE model on the test dataset, you can use the following command. Run this command from the home directory of the project:

```bash
python test_fge.py --models_path your_checkpoint_directory --img_path your_test_image_path
```
The result of testing is saved in the **results** directory. Curretly, I have not added the code to add the class name on the bounding box. You can add the class name by modifying the code in the **test_fge.py** file.

## SWA Implementation with Cyclical Averaging
If you want to train the model using the SWA method, you can use the following command. Run this command from the **swa** direcotry:

```bash
python train_swa.py --data_path your_dataset.yaml --model_path your_custom_model --save_dir checkpoint_directory --epochs 300 --base_epoch 258 --batch_size 16 --image_size 416 --base_lr 0.0001 --max_lr 0.01
```

The arguments to pass duiing training are:
* `--data_path` : Path to the dataset.yaml file (**dataset.yaml**). Add your own directory or the code defaults to **coco128.yaml**.
* `--model_path` : Path to the custom model. You can add your custom model. The code defaults to **yolov8l.pt**.
* `--save_dir` : Directory to save the checkpoints. Add your own directory or the code defaults to **fge_checkpoints**.
* `--epochs` : Number of epochs to train the model. The code defaults to **300**.
* `--base_epoch` : The epoch number to start the ensembling and save the base model. The code defaults to **258**.
* `--batch_size` : Batch size for training. The code defaults to **16**.
* `--image_size` : Image size for training. The code defaults to **416**.
* `--base_lr` : The lower learning rate for training. The code defaults to **0.0001**.
* `--max_lr` : The base higher learning rate for training. The code defaults to **0.01**.

The model checkpoints are saved in the directory passed in the argument.