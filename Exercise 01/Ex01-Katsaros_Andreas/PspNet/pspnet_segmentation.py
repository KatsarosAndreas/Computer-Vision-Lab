# pspnet_eval.py

# Patch for Python 3.10+ (collections.Iterable -> collections.abc.Iterable),
# ensuring compatibility for data loading and model usage
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable

# Mount Google Drive to access the given dataset and model checkpoints
from google.colab import drive
drive.mount('/content/drive')

#Import necessary libraries for CNN, SPP (pyramids), training & evaluation
import sys, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#Define paths to codes and dataset in Drive
base_dir = '/content/drive/My Drive/CV_1-PYRAMIDS'
codes_dir = os.path.join(base_dir, 'codes_and_models')
dataset_dir = os.path.join(base_dir, 'cityscapes_dataset')

# Check contents, helpful for verifying correct paths
print("Contents of codes directory:")
print(os.listdir(codes_dir))
print("\nContents of dataset directory:")
print(os.listdir(dataset_dir))

#Append codes_dir to sys.path to import custom modules (PSPNet, dataset)
sys.path.append(codes_dir)

from pspnet import PSPNet           #  PSPNet class (pyramids for CNN)
from cityscapes_dataset import Cityscapes  # Cityscapes dataset class

#Load the trained PSPNet model (which uses pyramidal pooling)
checkpoint_path = os.path.join(codes_dir, 'train_epoch_200_CPU.pth')
num_classes = 35
model = PSPNet(35)  #PSPNet with 35 classes

#Overwrite final conv layers to match the 35 classes in the checkpoint
model.cls[4] = nn.Conv2d(512, num_classes, kernel_size=1)
if hasattr(model, 'aux'):
    model.aux[4] = nn.Conv2d(256, num_classes, kernel_size=1)

#Load the checkpoint on CPU (pretrained weights)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()  # Set model to eval mode for inference

#Move model to GPU if available, helps with faster inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Prepare Cityscapes validation dataset to evaluate the model performance
val_list_path = os.path.join(dataset_dir, 'list', 'cityscapes', 'fine_val.txt')
val_dataset = Cityscapes(split='val', data_root=dataset_dir, data_list=val_list_path)

#Mean/std used in training, needed for proper normalization
mean = [0.485*255, 0.456*255, 0.406*255]
std  = [0.229*255, 0.224*255, 0.225*255]

# Validation transform with cropping + normalization,
# matching the input constraints for PSPNet
import transform as tfs
val_dataset.transform = tfs.Compose([
    tfs.Crop([713, 713], crop_type='center', padding=mean, ignore_label=255),
    tfs.ToTensor(),
    tfs.Normalize(mean=mean, std=std)
])

#Fix any incorrect image paths if needed (sometimes colab path mismatch)
for i, (img_path, label_path) in enumerate(val_dataset.data_list):
    if not os.path.exists(img_path):
        if "leftImg8bit/val/" in img_path and "leftImg8bit/val/val/" not in img_path:
            new_img_path = img_path.replace("leftImg8bit/val/", "leftImg8bit/val/val/")
            if os.path.exists(new_img_path):
                print(f"Fixed image path for sample {i}")
                val_dataset.data_list[i] = (new_img_path, label_path)

# Quick check on first sample path
sample_image_path, sample_label_path = val_dataset.data_list[0]
print("First image path (fixed):", sample_image_path, "exists?", os.path.exists(sample_image_path))
print("First label path:", sample_label_path, "exists?", os.path.exists(sample_label_path))

# Create DataLoader for the validation set (cityscapes images)
from torch.utils.data import DataLoader
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

#Define function to compute IoU (common metric for segmentation tasks)
def compute_iou(pred, target, num_classes=num_classes):
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

# Evaluate the model on each sample: compute class-wise IoU
ious_per_class = {cls: [] for cls in range(num_classes)}
mean_iou_per_image = []
num_samples = len(val_dataset)
print(f"Evaluating on {num_samples} validation samples...")

for idx in range(num_samples):
    try:
        image, gt_mask = val_dataset[idx]
    except Exception as e:
        print(f"Skipping sample {idx} due to error: {e}")
        continue

    #Inference: forward pass
    image = image.unsqueeze(0).to(device)
    output = model(image)
    pred_mask = torch.argmax(output, dim=1)[0]

    #Per-class IoU
    ious = compute_iou(pred_mask, gt_mask, num_classes=num_classes)
    for cls in range(num_classes):
        ious_per_class[cls].append(ious[cls])
    mean_iou_per_image.append(np.nanmean(ious))

    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1} samples...")

with open(os.path.join(codes_dir, 'cityscapes_names.txt'), 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

print("\nPer-class IoU:")
for cls in range(num_classes):
    valid = [iou for iou in ious_per_class[cls] if not np.isnan(iou)]
    if valid:
        mean_cls = np.mean(valid)
        std_cls = np.std(valid)
        print(f"{class_names[cls]}: Mean IoU = {mean_cls:.4f}, Std = {std_cls:.4f}")
    else:
        print(f"{class_names[cls]}: No valid pixels in ground truth.")

overall_mean = np.mean(mean_iou_per_image)
overall_std = np.std(mean_iou_per_image)
print(f"\nOverall mean IoU: Mean = {overall_mean:.4f}, Std = {overall_std:.4f}")
