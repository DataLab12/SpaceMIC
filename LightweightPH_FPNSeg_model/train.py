import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
import sys
import os
from torch.amp import autocast, GradScaler
import math
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
import Lightweight_PH_FPNSeg as Model
import dataset
import evaluation_metrics as mt

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/cm/shared/apps/cuda11.8/toolkit/11.8.0"

if "XLA_FLAGS" in os.environ:
    del os.environ["XLA_FLAGS"]

# Verify CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() == 0:
    raise RuntimeError("No CUDA devices found")
torch.cuda.empty_cache()  # Clear GPU memory

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Using device: {device}")


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def plot_validation_images(images, gt_masks, pred_masks, epoch, fold, boxes=None, 
                           output_dir="./val_plots"):
    os.makedirs(output_dir, exist_ok=True)
    num_images = min(images.size(0), 3)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = [axes]
    for i in range(num_images):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        gt = gt_masks[i].cpu().numpy()
        pred = (pred_masks[i].cpu().numpy() > 0.5).astype(np.uint8)
        axes[i][0].imshow(img)
        if boxes and len(boxes[i]) > 0 and boxes[i][0, 0] != 0:
            for box in boxes[i]:
                x_min, y_min, x_max, y_max = box.cpu().numpy()
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                axes[i][0].add_patch(rect)
        axes[i][0].set_title("Input Image with Boxes")
        axes[i][0].axis("off")
        axes[i][1].imshow(gt, cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")
        axes[i][2].imshow(pred, cmap="gray")
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"fold_{fold+1}_epoch_{epoch+1}_val_images.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved validation plot: {plot_path}")



# Training loop
num_epochs = 250
batch_size = 8
kfold = KFold(n_splits=3, shuffle=True, random_state=42)
if device.type == 'cuda':
    torch.cuda.empty_cache()
total_start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset.train_pairs)):
    fold_start_time = time.time()
    print(f"\n--- Fold {fold+1} ---")
    try:
        train_fold_pairs = [dataset.train_pairs[i] for i in train_idx]
        val_fold_pairs = [dataset.train_pairs[i] for i in val_idx]
        print(f"Train set size: {len(train_fold_pairs)}, Val set size: {len(val_fold_pairs)}")
        train_subset = dataset.CorrosionDataset(train_fold_pairs, dataset.train_image_dir, dataset.train_mask_dir, transform=dataset.train_transform)
        val_subset = dataset.CorrosionDataset(val_fold_pairs, dataset.train_image_dir, dataset.train_mask_dir, transform=dataset.val_transform)
        train_loader = dataset.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dataset.custom_collate_fn)
        val_loader = dataset.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.custom_collate_fn)
        print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")
        model = Model.SegmentationModel(in_chans=3).to(device)
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        model.apply(init_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / 5.0 if epoch < 5 else 1.0)
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 5, eta_min=1e-6)
        criterion = mt.BCEDiceLoss()
        scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
        best_val_dice = 0.0
        best_model_path = f"./best_corrosion_model_fold{fold+1}.pth"
        patience = 25
        counter = 0
        metrics_file = f"./metrics_fold{fold+1}.csv"
        metrics = []
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            model.train()
            train_loss, train_bce, train_dice_loss, train_l2, train_dice, train_iou = 0, 0, 0, 0, 0, 0
            train_batches = 0
            for i, batch in enumerate(tqdm(train_loader, desc="Training")):
                try:
                    image = batch["image"].to(device, non_blocking=True)
                    mask_gt = batch["mask"].to(device, non_blocking=True, dtype=torch.float32)
                    boxes = [b.to(device, non_blocking=True) for b in batch["boxes"]]
                    labels = [l.to(device, non_blocking=True) for l in batch["labels"]]
                    with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                        mask_logits = model(image, boxes, labels).squeeze(1)
                        loss_dict = criterion(mask_logits, mask_gt, model.parameters())
                        batch_dice = mt.dice_score(mask_logits, mask_gt)
                        batch_iou = mt.iou_score(mask_logits, mask_gt)
                    total_loss = loss_dict['total_loss']
                    if not (torch.isfinite(mask_logits).all() and torch.isfinite(total_loss)):
                        logger.warning(f"Non-finite values in batch {i}")
                        continue
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    train_loss += total_loss.item()
                    train_bce += loss_dict['bce_loss'].item()
                    train_dice_loss += loss_dict['dice_loss'].item()
                    train_l2 += loss_dict['l2_loss'].item()
                    train_dice += batch_dice
                    train_iou += batch_iou
                    train_batches += 1
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error in training batch {i}: {str(e)}")
                    continue
            scheduler.step() if epoch < 5 else scheduler_cosine.step()
            model.eval()
            val_loss, val_bce, val_dice_loss, val_l2, val_dice, val_iou = 0, 0, 0, 0, 0, 0
            val_batches = 0
            plot_batches = []
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    try:
                        image = batch["image"].to(device, non_blocking=True)
                        mask_gt = batch["mask"].to(device, non_blocking=True, dtype=torch.float32)
                        boxes = [b.to(device, non_blocking=True) for b in batch["boxes"]]
                        labels = [l.to(device, non_blocking=True) for l in batch["labels"]]
                        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                            mask_logits = model(image, boxes, labels).squeeze(1)
                            loss_dict = criterion(mask_logits, mask_gt, model.parameters())
                            batch_dice = mt.dice_score(mask_logits, mask_gt)
                            batch_iou = mt.iou_score(mask_logits, mask_gt)
                        val_loss += loss_dict['total_loss'].item()
                        val_bce += loss_dict['bce_loss'].item()
                        val_dice_loss += loss_dict['dice_loss'].item()
                        val_l2 += loss_dict['l2_loss'].item()
                        val_dice += batch_dice
                        val_iou += batch_iou
                        val_batches += 1
                        if (epoch + 1) % 5 == 0 and len(plot_batches) < 3:
                            plot_batches.append((image, mask_gt, torch.sigmoid(mask_logits), boxes))
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Error in validation batch {i}: {str(e)}")
                        continue
            if (epoch + 1) % 5 == 0 and plot_batches:
                for img, gt, pred, boxes in plot_batches:
                    try:
                        plot_validation_images(img, gt, pred, epoch, fold, boxes=boxes)
                    except Exception as e:
                        logger.error(f"Error plotting validation images: {str(e)}")
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            avg_train_bce = train_bce / train_batches if train_batches > 0 else 0.0
            avg_train_dice_loss = train_dice_loss / train_batches if train_batches > 0 else 0.0
            avg_train_l2 = train_l2 / train_batches if train_batches > 0 else 0.0
            avg_train_dice = train_dice / train_batches if train_batches > 0 else 0.0
            avg_train_iou = train_iou / train_batches if train_batches > 0 else 0.0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            avg_val_bce = val_bce / val_batches if val_batches > 0 else 0.0
            avg_val_dice_loss = val_dice_loss / val_batches if val_batches > 0 else 0.0
            avg_val_l2 = val_l2 / val_batches if val_batches > 0 else 0.0
            avg_val_dice = val_dice / val_batches if val_batches > 0 else 0.0
            avg_val_iou = val_iou / val_batches if val_batches > 0 else 0.0
            print(f"Fold {fold+1} | Epoch {epoch+1}")
            print(f"Train Metrics | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f} | Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Dice: {avg_train_dice_loss:.4f})")
            print(f"Val Metrics   | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f} | Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce:.4f}, Dice: {avg_val_dice_loss:.4f})")
            metrics.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_dice': avg_train_dice,
                'val_dice': avg_val_dice,
                'train_iou': avg_train_iou,
                'val_iou': avg_val_iou
            })
            pd.DataFrame(metrics).to_csv(metrics_file, index=False)
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                counter = 0
                try:
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved for fold {fold+1} at epoch {epoch+1} (Dice: {best_val_dice:.4f})")
                except Exception as e:
                    logger.error(f"Error saving model: {str(e)}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        try:
            df = pd.read_csv(metrics_file)
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', color='blue')
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Fold {fold+1} Loss')
            plt.legend()
            plt.grid(True)
            loss_plot_path = f"./plots/fold_{fold+1}_loss_plot.png"
            os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Saved loss plot: {loss_plot_path}")
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['train_dice'], label='Train Dice', marker='o', color='green')
            plt.plot(df['epoch'], df['val_dice'], label='Val Dice', marker='o', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Dice Score')
            plt.title(f'Fold {fold+1} Dice Score')
            plt.legend()
            plt.grid(True)
            dice_plot_path = f"./plots/fold_{fold+1}_dice_plot.png"
            plt.savefig(dice_plot_path)
            plt.close()
            print(f"Saved dice plot: {dice_plot_path}")
        except Exception as e:
            logger.error(f"Error plotting metrics for fold {fold+1}: {str(e)}")
        fold_duration = time.time() - fold_start_time
        print(f"Fold {fold+1} execution time: {format_time(fold_duration)}")
    except Exception as e:
        logger.error(f"Error in fold {fold+1}: {str(e)}")
total_duration = time.time() - total_start_time
print(f"Total execution time: {format_time(total_duration)}")

