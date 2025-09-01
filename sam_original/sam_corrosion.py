import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import KFold
import time
import monai
from monai.losses import DiceCELoss
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import torch.nn as nn

# Environment setup
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.8/toolkit/11.8.0"
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset validation
def validate_dataset(image_paths, mask_paths):
    assert len(image_paths) == len(mask_paths), f"Mismatch between number of images ({len(image_paths)}) and masks ({len(mask_paths)})"
    for img_path, mask_path in zip(image_paths, mask_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            assert img.size == mask.size, f"Size mismatch: {img_path} ({img.size}) vs {mask_path} ({mask.size})"
        except Exception as e:
            print(f"Error loading {img_path} or {mask_path}: {e}")
            raise
    print(f"Dataset validation passed for {len(image_paths)} image-mask pairs")

# Bounding box generation
def get_bounding_box(ground_truth_map):
    if np.any(ground_truth_map):
        y_indices, x_indices = np.where(ground_truth_map > 0)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        perturb_min = 5
        perturb_max = 20
        xmin = max(0, xmin - random.randint(perturb_min, perturb_max))
        xmax = min(ground_truth_map.shape[1], xmax + random.randint(perturb_min, perturb_max))
        ymin = max(0, ymin - random.randint(perturb_min, perturb_max))
        ymax = min(0 + ground_truth_map.shape[0], ymax + random.randint(perturb_min, perturb_max))
        return [xmin, ymin, xmax, ymax]
    return [0, 0, 512, 512]  

# Custom dataset
class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, image_augmentations=None, mask_augmentations=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.image_augmentations = image_augmentations
        self.mask_augmentations = mask_augmentations
        self.image_resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # Resize to 512x512
            image = self.image_resize(image)
            mask = self.mask_resize(mask)

            # Apply augmentations 
            if self.image_augmentations and self.mask_augmentations:
                seed = np.random.randint(123)  # Same seed for consistency
                random.seed(seed)
                torch.manual_seed(seed)

                try:
                    image = self.image_augmentations(image)
                except Exception as e:
                    print(f"Augmentation error for image {image_path}: {e}")
                    fallback_augs = transforms.Compose([
                        t for t in self.image_augmentations.transforms
                        if not isinstance(t, transforms.ColorJitter)
                    ])
                    image = fallback_augs(image)

                random.seed(seed)
                torch.manual_seed(seed)
                try:
                    mask = self.mask_augmentations(mask)
                except Exception as e:
                    print(f"Augmentation error for mask {mask_path}: {e}")
                  

            mask_array = np.array(mask)
            mask = (mask_array > 127).astype(np.float32)  # Ensure binary mask

            processed_inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
            image_tensor = processed_inputs['pixel_values'].squeeze(0)
            mask_tensor = processed_inputs['labels'].squeeze(0)

            bbox = get_bounding_box(mask_tensor.numpy())
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

            return {
                'image': image_tensor,
                'bbox': bbox_tensor,
                'labels': mask_tensor,
                'filename': os.path.basename(image_path), 
            }
        except Exception as e:
            print(f"Error in __getitem__ at index {idx} ({self.image_paths[idx]}): {e}")
            raise

# Augmentation setup
image_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.GaussianBlur(kernel_size=3)
])

mask_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=transforms.InterpolationMode.NEAREST)
])

validation_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

# Data paths
base_dir = '../MIC_NASA_dataset/'
data_paths = {
    'train': {
        'images': sorted(glob.glob(os.path.join(base_dir, 'train', 'images', '*.png'))),
        'labels': sorted(glob.glob(os.path.join(base_dir, 'train', 'masks', '*.png')))
    },
    'test': {
        'images': sorted(glob.glob(os.path.join(base_dir, 'test', 'images', '*.png'))),
        'labels': sorted(glob.glob(os.path.join(base_dir, 'test', 'masks', '*.png')))
    }
}

# Validate datasets
validate_dataset(data_paths['train']['images'], data_paths['train']['labels'])
validate_dataset(data_paths['test']['images'], data_paths['test']['labels'])

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Image unnormalization for visualization
def unnormalize_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# Visualize example
train_dataset_aug = SAMDataset(
    image_paths=data_paths['train']['images'],
    mask_paths=data_paths['train']['labels'],
    processor=processor,
    image_augmentations=image_augmentations,
    mask_augmentations=mask_augmentations
)

try:
    example = train_dataset_aug[5]
    image_display = unnormalize_image(example['image'])
    mask_display = example['labels'].squeeze().cpu().numpy()
    xmin, ymin, xmax, ymax = get_bounding_box(example['labels'].numpy())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image_display)
    axs[0].axis('off')
    axs[1].imshow(mask_display, cmap='gray')
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
    axs[1].add_patch(rect)
    axs[1].axis('off')
    plt.tight_layout()

    output_folder = "corrosion_aug_images"
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "example_5.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved output to {save_path}")
except Exception as e:
    print(f"Error visualizing example: {e}")

# Model setup
def My_Model():
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    return model.to(device)

# Evaluation metrics 
def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Dice loss
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    N = pred.size(0)
    pred_flat = pred.view(N, -1)
    target_flat = target.view(N, -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
    return 1 - dice.mean()


def calculate_accuracy(predicted, target):
    correct = (predicted == target).float().sum()
    total = target.numel()
    return correct / total

def sensitivity_recall(predicted, target):
    tp = ((predicted == 1) & (target == 1)).sum()
    fn = ((predicted == 0) & (target == 1)).sum()
    recall = tp / (tp + fn + 1e-6)
    return recall

def specificity_score(predicted, target):
    tn = ((predicted == 0) & (target == 0)).sum()
    fp = ((predicted == 1) & (target == 0)).sum()
    specificity = tn / (tn + fp + 1e-6)
    return specificity

def precision_score(predicted, target):
    tp = ((predicted == 1) & (target == 1)).sum()
    fp = ((predicted == 1) & (target == 0)).sum()
    precision = tp / (tp + fp + 1e-6)
    return precision

def matthews_correlation_coefficient(predicted, target):
    tp = ((predicted == 1) & (target == 1)).sum().float()
    tn = ((predicted == 0) & (target == 0)).sum().float()
    fp = ((predicted == 1) & (target == 0)).sum().float()
    fn = ((predicted == 0) & (target == 1)).sum().float()
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).float().sqrt() + 1e-6
    mcc = numerator / denominator
    return mcc if not torch.isnan(mcc) and not torch.isinf(mcc) else torch.tensor(0.0)

# Loss Function
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7, l2_weight=1e-8,
                 pos_weight=20.0, use_focal=True, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.l2_weight = float(l2_weight)
        self.use_focal = bool(use_focal)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def focal_loss(self, pred, target):
        target = target.to(dtype=pred.dtype)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce
        return loss.mean()

    def forward(self, pred, target, parameters=None):
        target = target.to(dtype=pred.dtype)

        # BCE / Focal
        if self.use_focal:
            bce = self.focal_loss(pred, target)
        else:
            bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)

        # Differentiable soft Dice 
        dloss = dice_loss(pred, target)

        # L2 
        if parameters is not None and self.l2_weight > 0:
            params = [p for p in parameters if p.requires_grad]
            if len(params) > 0:
                l2_sum = sum(p.pow(2.0).sum() for p in params)
                l2 = l2_sum * self.l2_weight
            else:
                l2 = torch.zeros((), device=pred.device, dtype=pred.dtype)
        else:
            l2 = torch.zeros((), device=pred.device, dtype=pred.dtype)

        total_loss = self.bce_weight * bce + self.dice_weight * dloss + l2

        return {
            'total_loss': total_loss,
            'bce_loss': bce,
            'dice_loss': dloss,
            'l2_loss': l2
        }

# Training setup
num_epochs = 250
num_folds = 3
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
patience = 25

fold_performance = {}
fold_times = {}

# Clear CUDA memory
torch.cuda.empty_cache()
import gc
gc.collect()

for fold, (train_idx, val_idx) in enumerate(kfold.split(data_paths['train']['images'])):
    print(f'Fold {fold+1}/{num_folds}')
    start_time = time.time()

    train_dataset = SAMDataset(
        image_paths=[data_paths['train']['images'][i] for i in train_idx],
        mask_paths=[data_paths['train']['labels'][i] for i in train_idx],
        processor=processor,
        image_augmentations=image_augmentations,
        mask_augmentations=mask_augmentations
    )
    val_dataset = SAMDataset(
        image_paths=[data_paths['train']['images'][i] for i in val_idx],
        mask_paths=[data_paths['train']['labels'][i] for i in val_idx],
        processor=processor,
        image_augmentations=validation_augmentations,
        mask_augmentations=validation_augmentations
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = My_Model()

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    seg_loss = BCEDiceLoss()
    best_val_dice = 0.0
    patience_counter = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses, train_dices, train_ious, train_dice_losses = [], [], [], []

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            try:
                optimizer.zero_grad()
                pixel_values = batch["image"].to(device)
                input_boxes = batch["bbox"].unsqueeze(1).to(device)
                ground_truth_masks = batch["labels"].to(device)
                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                predicted_masks = outputs['pred_masks'].squeeze(1)

                loss_dict = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1).float(),
                                     parameters=model.parameters())
                loss = loss_dict['total_loss']
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # Metrics 
                with torch.no_grad():
                    pred_probs = torch.sigmoid(predicted_masks)
                    true_masks = ground_truth_masks.unsqueeze(1).float()
                    dice_val = dice_coefficient(pred_probs, true_masks).item()
                    iou_val = iou_score(pred_probs, true_masks).item()
                    dice_metric = dice_loss(predicted_masks, true_masks).item()  # soft dice loss value
                    train_dices.append(dice_val)
                    train_ious.append(iou_val)
                    train_dice_losses.append(dice_metric)
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        model.eval()
        val_losses, val_dices, val_ious, val_dice_losses = [], [], [], []
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                try:
                    pixel_values = val_batch["image"].to(device)
                    input_boxes = val_batch["bbox"].unsqueeze(1).to(device)
                    ground_truth_masks = val_batch["labels"].to(device)
                    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                    predicted_val_masks = outputs['pred_masks'].squeeze(1)

                    val_loss_dict = seg_loss(predicted_val_masks, ground_truth_masks.unsqueeze(1).float(),
                                             parameters=model.parameters())
                    val_losses.append(val_loss_dict['total_loss'].item())

                    pred_probs = torch.sigmoid(predicted_val_masks)
                    true_masks = ground_truth_masks.unsqueeze(1).float()
                    dice_val = dice_coefficient(pred_probs, true_masks).item()
                    iou_val = iou_score(pred_probs, true_masks).item()
                    dice_metric = dice_loss(predicted_val_masks, true_masks).item()
                    val_dices.append(dice_val)
                    val_ious.append(iou_val)
                    val_dice_losses.append(dice_metric)
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        avg_train_dice = np.mean(train_dices) if train_dices else float('nan')
        avg_val_dice = np.mean(val_dices) if val_dices else float('nan')
        avg_train_iou = np.mean(train_ious) if train_ious else float('nan')
        avg_val_iou = np.mean(val_ious) if val_ious else float('nan')
        avg_train_dice_loss = np.mean(train_dice_losses) if train_dice_losses else float('nan')
        avg_val_dice_loss = np.mean(val_dice_losses) if val_dice_losses else float('nan')

        print(
            f'Epoch [{epoch+1}/{num_epochs}] | '
            f'Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, IoU: {avg_train_iou:.4f}, Dice Loss: {avg_train_dice_loss:.4f} || '
            f'Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}, Dice Loss: {avg_val_dice_loss:.4f}'
        )

        if val_dices and avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), f"./best_model_SAM_fold_{fold+1}.pth")
            print(f"New best model saved at epoch {epoch+1} with Val Dice: {best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement. Best epoch: {best_epoch+1}')
            break

    end_time = time.time()
    elapsed = end_time - start_time
    fold_times[fold] = elapsed
    fold_performance[fold] = best_val_dice

    print(f"Fold {fold+1} training time: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")

for fold, performance in fold_performance.items():
    print(f'Fold {fold+1}: Best Validation Dice: {performance:.4f} | Training Time: {fold_times[fold]/60:.2f} min')

total_time = sum(fold_times.values())
print(f"\nTotal training time for {num_folds} folds: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")

# Test evaluation 
test_dataset = SAMDataset(
    image_paths=data_paths['test']['images'],
    mask_paths=data_paths['test']['labels'],
    processor=processor
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"Test dataset size: {len(test_dataset)}")

output_folder = "sam_test_outputs"
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        try:
            outputs = model(pixel_values=batch["image"].to(device),
                           input_boxes=batch["bbox"].unsqueeze(1).to(device),
                           multimask_output=False)

            predicted_probs = torch.sigmoid(outputs.pred_masks.squeeze(1))
            predicted_masks = (predicted_probs > 0.5).float()

            batch_size = batch["image"].shape[0]
            for j in range(batch_size):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                img = batch["image"][j]
                img_disp = unnormalize_image(img)
                axs[0].imshow(img_disp)
                axs[0].set_title('Input Image')
                axs[0].axis('off')

                gt_mask = batch["labels"][j].cpu().squeeze().numpy()
                axs[1].imshow(gt_mask, cmap='gray')
                axs[1].set_title('Ground Truth Mask')
                axs[1].axis('off')

                pred_mask = predicted_masks[j].cpu().squeeze().numpy()
                axs[2].imshow(pred_mask, cmap='gray')
                axs[2].set_title('Predicted Mask')
                axs[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(output_folder, f"test_batch{i}_img{j}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
        except Exception as e:
            print(f"Error in test batch {i}: {e}")
            continue

# Test metrics
total_dice = total_iou = total_accuracy = total_sensitivity = total_specificity = 0
total_precision = total_mcc = total_auc = total_mse = 0
num_samples = 0

with torch.no_grad():
    for batch in tqdm(test_loader):
        try:
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=batch["image"].to(device),
                           input_boxes=batch["bbox"].unsqueeze(1).to(device),
                           multimask_output=False)
            predicted_probs = torch.sigmoid(outputs['pred_masks'].squeeze(1))
            predicted_masks = (predicted_probs > 0.5).float().squeeze(1)

            probs_flat = predicted_probs.detach().cpu().flatten().numpy()
            labels_flat = labels.detach().cpu().flatten().numpy()
            try:
                auc = roc_auc_score(labels_flat, probs_flat)
            except ValueError:
                auc = float('nan')

            mse = F.mse_loss(predicted_probs, labels.float())

            dice_score = dice_coefficient(predicted_masks, labels)
            iou = iou_score(predicted_masks, labels)
            accuracy = calculate_accuracy(predicted_masks, labels)
            sensitivity = sensitivity_recall(predicted_masks, labels)
            specificity = specificity_score(predicted_masks, labels)
            precision = precision_score(predicted_masks, labels)
            mcc = matthews_correlation_coefficient(predicted_masks, labels)

            print(f"Dice: {dice_score.item():.4f} | IoU: {iou.item():.4f} | Acc: {accuracy.item():.4f} | "
                  f"Sen: {sensitivity.item():.4f} | Spec: {specificity.item():.4f} | Prec: {precision.item():.4f} | "
                  f"MCC: {mcc.item():.4f} | AUC: {auc:.4f} | MSE: {mse.item():.4f}")

            total_dice += dice_score.item()
            total_iou += iou.item()
            total_accuracy += accuracy.item()
            total_sensitivity += sensitivity.item()
            total_specificity += specificity.item()
            total_precision += precision.item()
            total_mcc += mcc.item()
            total_auc += auc if not np.isnan(auc) else 0
            total_mse += mse.item()
            num_samples += 1
        except Exception as e:
            print(f"Error in test batch: {e}")
            continue

print(f"\nAverage Dice: {total_dice/num_samples:.4f}")
print(f"Average IoU: {total_iou/num_samples:.4f}")
print(f"Average Accuracy: {total_accuracy/num_samples:.4f}")
print(f"Average Sensitivity (Recall): {total_sensitivity/num_samples:.4f}")
print(f"Average Specificity: {total_specificity/num_samples:.4f}")
print(f"Average Precision: {total_precision/num_samples:.4f}")
print(f"Average MCC: {total_mcc/num_samples:.4f}")
print(f"Average AUC: {total_auc/num_samples:.4f}")
print(f"Average MSE: {total_mse/num_samples:.4f}")

