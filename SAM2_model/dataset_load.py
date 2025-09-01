import logging
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.ndimage import label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MIC_NASA_dataset'))

train_image_dir = os.path.join(base_dir, 'train', 'images')
train_mask_dir  = os.path.join(base_dir, 'train', 'masks')
test_image_dir  = os.path.join(base_dir, 'test',  'images')
test_mask_dir   = os.path.join(base_dir, 'test',  'masks')

for directory in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

train_pairs = []
test_pairs = []
try:
    train_image_files = sorted([f for f in os.listdir(train_image_dir) if f.endswith(('.png', '.tif'))])
    train_mask_files = sorted([f for f in os.listdir(train_mask_dir) if f.endswith(('.png', '.tif'))])
    test_image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.tif'))])
    test_mask_files = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(('.png', '.tif'))])
    logger.info(f"Found {len(train_image_files)} train images, {len(test_image_files)} test images")
    for img_file in train_image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        if mask_file in train_mask_files:
            train_pairs.append((img_file, mask_file))
    for img_file in test_image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        if mask_file in test_mask_files:
            test_pairs.append((img_file, mask_file))
except Exception as e:
    logger.error(f"Error reading directories: {e}")
    raise
logger.info(f"Total train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")
if not train_pairs or not test_pairs:
    raise ValueError("Dataset is empty")

class CorrosionDataset(Dataset):
    def __init__(self, paired_filenames, image_dir, mask_dir, target_size=(512, 512), transform=None):
        self.paired_filenames = paired_filenames
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.paired_filenames)

    def __getitem__(self, idx):
        image_file, mask_file = self.paired_filenames[idx]
        try:
            image_path = os.path.join(self.image_dir, image_file)
            image = Image.open(image_path).convert("L").resize(self.target_size, Image.BILINEAR)
            image = np.array(image)
            image = np.stack([image] * 3, axis=-1)

            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = Image.open(mask_path).convert("L").resize(self.target_size, Image.NEAREST)
            mask = np.array(mask)
            mask = (mask > 128).astype(np.uint8)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"].float()
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).float()

            # Generate pointer prompt
            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            points = []
            region_labels = []
            if mask_np.sum() > 0:
                labeled_mask, num_regions = label(mask_np)
                for region_id in range(1, num_regions + 1):
                    region = (labeled_mask == region_id)
                    coords = np.argwhere(region)
                    if coords.size > 0:
                        # Use centroid as pointer
                        y_mean = int(coords[:, 0].mean())
                        x_mean = int(coords[:, 1].mean())
                        points.append([x_mean, y_mean])
                        region_labels.append(1)
                # Add a background point
                bg_coords = np.argwhere(mask_np == 0)
                if len(bg_coords) > 0:
                    bg_idx = np.random.choice(len(bg_coords))
                    points.append([bg_coords[bg_idx][1], bg_coords[bg_idx][0]])
                    region_labels.append(0)
            else:
                points = [[self.target_size[1] // 2, self.target_size[0] // 2]]
                region_labels = [0]

            points = np.array(points)
            region_labels = np.array(region_labels)

            return {
                "image": image,
                "mask": mask,
                "points": torch.from_numpy(points).float(),   
                "labels": torch.from_numpy(region_labels).float(), 
                "image_file": image_file,
                "mask_file": mask_file
            }
        except Exception as e:
            logger.error(f"Error loading {image_file}/{mask_file}: {str(e)}")
            raise

train_transform = A.Compose([
    A.HorizontalFlip(p=0.6),  
    A.VerticalFlip(p=0.4),  
    A.RandomBrightnessContrast(
        p=0.7, 
        brightness_limit=0.3,  
        contrast_limit=0.3 
    ),
    A.Rotate(limit=30, p=0.5), 
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=15, 
        p=0.5 
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  
    A.RandomGamma(gamma_limit=(80, 120), p=0.3), 
    A.OneOf([ 
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.5),
    ], p=0.3),
    A.HueSaturationValue(
        hue_shift_limit=20, 
        sat_shift_limit=30, 
        val_shift_limit=20, 
        p=0.2 
    ),

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

