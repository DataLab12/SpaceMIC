import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label
from PIL import Image


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'MIC_NASA_dataset'))

train_image_dir = os.path.join(base_dir, 'train', 'images')
train_mask_dir  = os.path.join(base_dir, 'train', 'masks')
test_image_dir  = os.path.join(base_dir, 'test',  'images')
test_mask_dir   = os.path.join(base_dir, 'test',  'masks')

for directory in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")

train_pairs = []
test_pairs = []
try:
    train_image_files = sorted([f for f in os.listdir(train_image_dir) if f.endswith(('.png', '.tif'))])
    train_mask_files = sorted([f for f in os.listdir(train_mask_dir) if f.endswith(('.png', '.tif'))])
    test_image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.png', '.tif'))])
    test_mask_files = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(('.png', '.tif'))])
    print(f"Found {len(train_image_files)} train images, {len(test_image_files)} test images")
    for img_file in train_image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        if mask_file in train_mask_files:
            train_pairs.append((os.path.join(train_image_dir, img_file), os.path.join(train_mask_dir, mask_file)))
    for img_file in test_image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = f"{base_name}.png"
        if mask_file in test_mask_files:
            test_pairs.append((os.path.join(test_image_dir, img_file), os.path.join(test_mask_dir, mask_file)))
except Exception as e:
    print(f"Error reading directories: {e}")
    raise
print(f"Total train pairs: {len(train_pairs)}, test pairs: {len(test_pairs)}")
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

            mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask
            if mask_np.sum() > 0:
                labeled_mask, num_regions = label(mask_np)
                boxes = []
                region_labels = []
                for region_id in range(1, num_regions + 1):
                    region = (labeled_mask == region_id)
                    coords = np.where(region)
                    if len(coords[0]) > 0:
                        y_min = int(np.min(coords[0]))
                        y_max = int(np.max(coords[0]))
                        x_min = int(np.min(coords[1]))
                        x_max = int(np.max(coords[1]))
                        boxes.append([x_min, y_min, x_max, y_max])
                        region_labels.append(1)
                boxes = np.array(boxes) if boxes else np.array([[0, 0, 0, 0]])
                region_labels = np.array(region_labels) if region_labels else np.array([0])
            else:
                boxes = np.array([[0, 0, 0, 0]])
                region_labels = np.array([0])

            if len(region_labels) > 0 and region_labels[0] == 1:
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.target_size[1] - 1)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.target_size[0] - 1)

            if len(region_labels) > 0 and region_labels[0] == 1 and mask_np.sum() == 0:
                print(f"Label=1 but mask is empty for {image_file}")
                boxes = np.array([[0, 0, 0, 0]])
                region_labels = np.array([0])
            elif len(region_labels) == 0 and mask_np.sum() > 0:
                print(f"Label=0 but mask has foreground for {image_file}")
                labeled_mask, num_regions = label(mask_np)
                boxes = []
                region_labels = []
                for region_id in range(1, num_regions + 1):
                    region = (labeled_mask == region_id)
                    coords = np.where(region)
                    if len(coords[0]) > 0:
                        y_min = int(np.min(coords[0]))
                        y_max = int(np.max(coords[0]))
                        x_min = int(np.min(coords[1]))
                        x_max = int(np.max(coords[1]))
                        boxes.append([x_min, y_min, x_max, y_max])
                        region_labels.append(1)
                boxes = np.array(boxes)
                region_labels = np.array(region_labels)

            return {
                "image": image,
                "mask": mask,
                "boxes": torch.from_numpy(boxes).float(),
                "labels": torch.from_numpy(region_labels).float(),
                "image_file": image_file,
                "mask_file": mask_file
            }
        except Exception as e:
            print(f"Error loading {image_file}/{mask_file}: {str(e)}")
            raise

def custom_collate_fn(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_file": [item["image_file"] for item in batch]
    }

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

# print(f"Training transform: {train_transform}")

# Create datasets
train_dataset = CorrosionDataset(train_pairs, train_image_dir, train_mask_dir, transform=train_transform)


