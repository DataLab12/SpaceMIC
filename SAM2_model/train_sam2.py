import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import matplotlib.pyplot as plt
import logging
import sys
import pandas as pd
import evaluation_metrics as mt
import dataset_load as dl

from sam2.build_sam import build_sam2

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to format time
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

# Function to plot validation images
def plot_validation_images(images, gt_masks, pred_masks, epoch, fold, 
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
        axes[i][0].set_title("Input Image")
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
    print(f"[INFO] Saved validation plot: {plot_path}")

# Function to plot metrics from CSV
def plot_metrics(fold, metrics_file, output_dir="./plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metrics_file)
    epochs = df['epoch']
    train_loss = df['train_loss']
    val_loss = df['val_loss']
    train_dice = df['train_dice']
    val_dice = df['val_dice']

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, val_loss, label='Val Loss', marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, f'fold_{fold}_loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"[INFO] Saved loss plot: {loss_plot_path}")

    # Plot Dice
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_dice, label='Train Dice', marker='o', color='green')
    plt.plot(epochs, val_dice, label='Val Dice', marker='o', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title(f'Fold {fold} Dice Score')
    plt.legend()
    plt.grid(True)
    dice_plot_path = os.path.join(output_dir, f'fold_{fold}_dice_plot.png')
    plt.savefig(dice_plot_path)
    plt.close()
    print(f"[INFO] Saved dice plot: {dice_plot_path}")


amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'

sam2_checkpoint = "./sam2_repo/checkpoints/sam2.1_hiera_large.pt"

print(f"[INFO] Using device: {device}")
num_epochs = 250
batch_size = 1
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# Start total execution timer
total_start_time = time.time()

# K-fold training loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(dl.train_pairs)):
    fold_start_time = time.time()
    print(f"\n--- Fold {fold+1} ---")
    try:
        train_fold_pairs = [dl.train_pairs[i] for i in train_idx]
        val_fold_pairs = [dl.train_pairs[i] for i in val_idx]
        logger.info(f"Train set size: {len(train_fold_pairs)}, Val set size: {len(val_fold_pairs)}")

        train_subset = dl.CorrosionDataset(
            train_fold_pairs,
            image_dir=dl.train_image_dir,
            mask_dir=dl.train_mask_dir,
            target_size=(512, 512),
            transform=dl.train_transform
        )
        val_subset = dl.CorrosionDataset(
            val_fold_pairs,
            image_dir=dl.train_image_dir,
            mask_dir=dl.train_mask_dir,
            target_size=(512, 512),
            transform=dl.val_transform
        )


        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=mt.custom_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=mt.custom_collate_fn)

        logger.info(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")


        # Initialize model components
        sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_checkpoint, device=device)
        image_encoder = sam2_model.image_encoder.to(device)
        prompt_encoder = sam2_model.sam_prompt_encoder.to(device)
        mask_decoder = sam2_model.sam_mask_decoder.to(device)
        conv_256_to_32 = nn.Conv2d(256, 32, kernel_size=1).to(device)
        conv_128_to_64 = nn.Conv2d(256, 64, kernel_size=1).to(device)

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        conv_256_to_32.apply(init_weights)
        conv_128_to_64.apply(init_weights)
        print(f"[INFO] Initialized model weights for fold {fold+1}")

        # Optimizer
        optimizer = torch.optim.AdamW(
            list(image_encoder.parameters()) +
            list(prompt_encoder.parameters()) +
            list(mask_decoder.parameters()) +
            list(conv_256_to_32.parameters()) +
            list(conv_128_to_64.parameters()),
            lr=1e-4  # Increased base LR
        )

        # Learning rate scheduler with warmup and cosine
        def warmup_lambda(epoch):
            if epoch < 5:
                return (epoch + 1) / 5.0
            return 1.0
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - 5, eta_min=1e-6)

        # Loss function
        criterion = mt.BCEDiceLoss()

        # Mixed precision scaler
        scaler = GradScaler(amp_device)

        # Training configuration
        best_val_dice = 0.0
        best_model_path = f"./best_sam_based_model_fold{fold+1}.pth"
        patience = 25
        counter = 0
        effective_batch_size = 8
        accumulation_steps = effective_batch_size // batch_size

        # Initialize metrics logging
        metrics_file = f"./metrics_fold{fold+1}.csv"
        metrics = []

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            image_encoder.train()
            prompt_encoder.train()
            mask_decoder.train()
            train_loss, train_bce, train_dice_loss, train_l2, train_dice, train_iou = 0, 0, 0, 0, 0, 0
            train_batches = 0

            for i, batch in enumerate(tqdm(train_loader, desc="Training")):
                try:
                    image = batch["image"].to(device, non_blocking=True)
                    mask_gt = batch["mask"].to(device, non_blocking=True, dtype=torch.float32)
                    points = batch["points"]
                    labels = batch["labels"]

                    with autocast(device_type=amp_device, enabled=True):
                        # Encoder
                        encoder_output = image_encoder(image)
                        image_features = encoder_output["vision_features"]
                        image_pe = encoder_output["vision_pos_enc"]
                        backbone_fpn = encoder_output["backbone_fpn"]

        
                        if isinstance(image_pe, list):
                            image_pe = image_pe[-1]

                        # High-resolution features
                        high_res_features = [
                            conv_256_to_32(backbone_fpn[0]),
                            conv_128_to_64(backbone_fpn[1])
                        ]

                        # Prompt encoding for batched points/labels
                        sparse_emb = []
                        dense_emb = []
                        for b in range(len(points)):
                            point_coords = points[b].to(device).unsqueeze(0)  # [1, N, 2]
                            point_labels = labels[b].to(device).unsqueeze(0)  # [1, N]
                            sp, de = prompt_encoder(
                                points=(point_coords, point_labels),
                                boxes=None,
                                masks=None
                            )
                            sparse_emb.append(sp)
                            dense_emb.append(de)
                        sparse_emb = torch.cat(sparse_emb, dim=0)
                        dense_emb = torch.cat(dense_emb, dim=0)

                        if dense_emb.shape[-2:] != image_features.shape[-2:]:
                            dense_emb = F.interpolate(
                                dense_emb, size=image_features.shape[-2:], mode="bilinear", align_corners=False
                            )

                        # Mask decoding
                        masks, iou_pred, sam_tokens_out, object_score_logits = mask_decoder(
                            image_embeddings=image_features,
                            image_pe=image_pe,
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb,
                            multimask_output=False,
                            repeat_image=False,
                            high_res_features=high_res_features
                        )
                        mask_logits = masks.squeeze(1)

                        # Upsample mask_logits to match mask_gt
                        mask_logits = F.interpolate(mask_logits.unsqueeze(1), size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                        mask_logits = mask_logits.squeeze(1)

                        # Loss computation
                        loss_dict = criterion(mask_logits, mask_gt)
                        batch_dice = mt.dice_score(mask_logits, mask_gt)
                        batch_iou = mt.iou_score(mask_logits, mask_gt)

                    total_loss = loss_dict['total_loss']

                    if not torch.isfinite(mask_logits).all():
                        print(f"[WARNING] Non-finite values in mask_logits for batch {i}")
                        continue

                    if not torch.isfinite(total_loss):
                        print(f"[WARNING] Non-finite loss encountered in batch {i}. Skipping batch.")
                        continue

                    scaler.scale(total_loss / accumulation_steps).backward()
                    if (i + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], max_norm=1.0)  # Increased max_norm
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

                except Exception as e:
                    print(f"[ERROR] Error in training batch {i}: {e}")
                    continue

            # Scheduler step
            if epoch < 5:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            # Validation
            image_encoder.eval()
            prompt_encoder.eval()
            mask_decoder.eval()
            val_loss, val_bce, val_dice_loss, val_l2, val_dice, val_iou = 0, 0, 0, 0, 0, 0
            val_batches = 0
            plot_batches = []

            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    try:
                        image = batch["image"].to(device, non_blocking=True)
                        mask_gt = batch["mask"].to(device, non_blocking=True, dtype=torch.float32)
                        points = batch["points"]
                        labels = batch["labels"]

                        with autocast(device_type=amp_device, enabled=True):
                            encoder_output = image_encoder(image)
                            image_features = encoder_output["vision_features"]
                            image_pe = encoder_output["vision_pos_enc"]
                            backbone_fpn = encoder_output["backbone_fpn"]

                            if isinstance(image_pe, list):
                                image_pe = image_pe[-1]

                            high_res_features = [
                                conv_256_to_32(backbone_fpn[0]),
                                conv_128_to_64(backbone_fpn[1])
                            ]

                            sparse_emb = []
                            dense_emb = []
                            for b in range(len(points)):
                                point_coords = points[b].to(device).unsqueeze(0)  # [1, N, 2]
                                point_labels = labels[b].to(device).unsqueeze(0)  # [1, N]
                                sp, de = prompt_encoder(
                                    points=(point_coords, point_labels),
                                    boxes=None,
                                    masks=None
                                )
                                sparse_emb.append(sp)
                                dense_emb.append(de)
                            sparse_emb = torch.cat(sparse_emb, dim=0)
                            dense_emb = torch.cat(dense_emb, dim=0)

                            if dense_emb.shape[-2:] != image_features.shape[-2:]:
                                dense_emb = F.interpolate(
                                    dense_emb, size=image_features.shape[-2:], mode="bilinear", align_corners=False
                                )

                            masks, iou_pred, sam_tokens_out, object_score_logits = mask_decoder(
                                image_embeddings=image_features,
                                image_pe=image_pe,
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb,
                                multimask_output=False,
                                repeat_image=False,
                                high_res_features=high_res_features
                            )
                            mask_logits = masks.squeeze(1)

                            mask_logits = F.interpolate(mask_logits.unsqueeze(1), size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
                            mask_logits = mask_logits.squeeze(1)

                            loss_dict = criterion(mask_logits, mask_gt)
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
                            plot_batches.append((image, mask_gt, torch.sigmoid(mask_logits)))

                    except Exception as e:
                        print(f"[ERROR] Error in validation batch {i}: {e}")
                        continue

            # Plot validation images every 5 epochs
            if (epoch + 1) % 5 == 0 and plot_batches:
                for idx, (img, gt, pred) in enumerate(plot_batches):
                    try:
                        plot_validation_images(img, gt, pred, epoch, fold)
                    except Exception as e:
                        print(f"[ERROR] Error plotting validation images: {e}")

            # Evaluation summary
            if train_batches > 0:
                avg_train_loss = train_loss / train_batches
                avg_train_bce = train_bce / train_batches
                avg_train_dice_loss = train_dice_loss / train_batches
                avg_train_l2 = train_l2 / train_batches
                avg_train_dice = train_dice / train_batches
                avg_train_iou = train_iou / train_batches
            else:
                avg_train_loss = avg_train_bce = avg_train_dice_loss = avg_train_l2 = avg_train_dice = avg_train_iou = 0.0
                print("[WARNING] No valid training batches processed")

            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                avg_val_bce = val_bce / val_batches
                avg_val_dice_loss = val_dice_loss / val_batches
                avg_val_l2 = val_l2 / val_batches
                avg_val_dice = val_dice / val_batches
                avg_val_iou = val_iou / val_batches
            else:
                avg_val_loss = avg_val_bce = avg_val_dice_loss = avg_val_l2 = avg_val_dice = avg_val_iou = 0.0
                print("[WARNING] No valid validation batches processed")

            print(f"\n[INFO] Fold {fold+1} | Epoch {epoch+1}")
            print(f"[INFO] Train Metrics | Dice: {avg_train_dice:.4f} | IoU: {avg_train_iou:.4f} | Loss: {avg_train_loss:.4f} (BCE: {avg_train_bce:.4f}, Dice: {avg_train_dice_loss:.4f})")
            print(f"[INFO] Val Metrics   | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f} | Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce:.4f}, Dice: {avg_val_dice_loss:.4f})")

            # Log metrics to CSV
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

            # Early stopping and model saving
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                counter = 0
                try:
                    torch.save({
                        'image_encoder': image_encoder.state_dict(),
                        'prompt_encoder': prompt_encoder.state_dict(),
                        'mask_decoder': mask_decoder.state_dict(),
                        'conv_256_to_32': conv_256_to_32.state_dict(),
                        'conv_128_to_64': conv_128_to_64.state_dict()
                    }, best_model_path)
                    print(f"[INFO] Best model saved for fold {fold+1} at epoch {epoch+1} (Dice: {best_val_dice:.4f})")
                except Exception as e:
                    print(f"[ERROR] Error saving model: {e}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"[INFO] Early stopping at epoch {epoch+1}")
                    break

        # Plot metrics for this fold
        try:
            plot_metrics(fold + 1, metrics_file)
        except Exception as e:
            print(f"[ERROR] Error plotting metrics for fold {fold+1}: {e}")

        # Log fold execution time
        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        print(f"[INFO] Fold {fold+1} execution time: {format_time(fold_duration)}")
    except Exception as e:
        print(f"[ERROR] Error in fold {fold+1}: {e}")
        continue

# Log total execution time
total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"[INFO] Total execution time for all folds: {format_time(total_duration)}")


