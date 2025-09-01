import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
import sys
import os
from torch.amp import autocast
import math
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import Lightweight_PH_FPNSeg as Model
import dataset
import evaluation_metrics as mt
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


# Evaluation Function
def plot_test_predictions(images, gt_masks, pred_masks, fold, batch_idx, boxes=None,
                         output_dir="./test_plots"):
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
        if boxes and len(boxes[i]) > 0 and boxes[i].sum() > 0:
            for box in boxes[i]:
                x_min, y_min, x_max, y_max = box.cpu().numpy().astype(int)
                if x_max > x_min and y_max > y_min:
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
                    axes[i][0].add_patch(rect)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")
        axes[i][1].imshow(gt, cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")
        axes[i][2].imshow(pred, cmap="gray")
        axes[i][2].set_title("Predicted Mask")
        axes[i][2].axis("off")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"fold_{fold+1}_batch_{batch_idx+1}_test_images.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved test plot: {plot_path}")


def evaluate_test_set(test_pairs, model_paths, test_image_dir, test_mask_dir, device, batch_size=1, target_size=(512, 512),
                     mask_output_dir="./test_predicted_masks"):
    
    total_start_time = time.time()
    print("Starting evaluation on test set")
    print(f"Device: {device}, Device type: {device.type}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    for path in model_paths:
        print(f"Model path exists: {os.path.exists(path)}")
    
    test_dataset = dataset.CorrosionDataset(test_pairs, test_image_dir, test_mask_dir, transform=dataset.val_transform, target_size=target_size)
    test_loader = dataset.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.custom_collate_fn)
    print(f"Test set size: {len(test_dataset)}, Test loader: {len(test_loader)} batches")
    
    # Initialize lists for all metrics, including accuracy
    all_dice_scores, all_iou_scores, all_sensitivity_scores, all_specificity_scores, all_f1_scores, all_auc_scores, all_accuracy_scores = [], [], [], [], [], [], []
    os.makedirs(mask_output_dir, exist_ok=True)
    
    preds_dict = defaultdict(list)
    
    for fold, model_path in enumerate(model_paths):
        print(f"\nEvaluating model from fold {fold+1}: {model_path}")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print("Creating SegmentationModel")
        model = Model.SegmentationModel(in_chans=3).to(device)
        try:
            print(f"Loading model weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            continue
        model.eval()
        
        # Initialize fold-specific lists for metrics
        fold_dice_scores, fold_iou_scores, fold_sensitivity_scores, fold_specificity_scores, fold_f1_scores, fold_auc_scores, fold_accuracy_scores = [], [], [], [], [], [], []
        plot_batches = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating fold {fold+1}")):
                try:
                    image = batch["image"].to(device, non_blocking=True)
                    mask_gt = batch["mask"].to(device, non_blocking=True, dtype=torch.float32)
                    boxes = [b.to(device, non_blocking=True) for b in batch["boxes"]]
                    labels = [l.to(device, non_blocking=True) for l in batch["labels"]]
                    image_files = batch["image_file"]
                    if mask_gt.sum() == 0:
                        logger.warning(f"Skipping batch {batch_idx}: empty ground truth masks")
                        continue
                    with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                        mask_logits = model(image, boxes, labels).squeeze(1)
                        if not torch.isfinite(mask_logits).all():
                            logger.warning(f"Non-finite predictions in batch {batch_idx}")
                            continue
                        batch_dice = mt.dice_score(mask_logits, mask_gt)
                        batch_iou = mt.iou_score(mask_logits, mask_gt)
                        batch_sensitivity = mt.sensitivity_score(mask_logits, mask_gt)
                        batch_specificity = mt.specificity_score(mask_logits, mask_gt)
                        batch_f1 = mt.f1_score(mask_logits, mask_gt)
                        batch_auc = mt.auc_score(mask_logits, mask_gt)
                        batch_accuracy = mt.accuracy_score(mask_logits, mask_gt)  # Compute accuracy
                    # Move metrics to CPU and convert to scalar
                    fold_dice_scores.append(batch_dice.cpu().item() if torch.is_tensor(batch_dice) else batch_dice)
                    fold_iou_scores.append(batch_iou.cpu().item() if torch.is_tensor(batch_iou) else batch_iou)
                    fold_sensitivity_scores.append(batch_sensitivity.cpu().item() if torch.is_tensor(batch_sensitivity) else batch_sensitivity)
                    fold_specificity_scores.append(batch_specificity.cpu().item() if torch.is_tensor(batch_specificity) else batch_specificity)
                    fold_f1_scores.append(batch_f1.cpu().item() if torch.is_tensor(batch_f1) else batch_f1)
                    fold_auc_scores.append(batch_auc.cpu().item() if torch.is_tensor(batch_auc) else batch_auc)
                    fold_accuracy_scores.append(batch_accuracy.cpu().item() if torch.is_tensor(batch_accuracy) else batch_accuracy)  # Store accuracy
                    probs = torch.sigmoid(mask_logits).cpu().numpy()
                    for i, image_file in enumerate(image_files):
                        preds_dict[image_file].append(probs[i])
                    if len(plot_batches) < 3:
                        plot_batches.append((image, mask_gt, torch.sigmoid(mask_logits), boxes))
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error in test batch {batch_idx}: {e}")
                    continue
            for idx, (img, gt, pred, boxes) in enumerate(plot_batches):
                try:
                    plot_test_predictions(img, gt, pred, fold, idx, boxes=boxes)
                except Exception as e:
                    logger.error(f"Error plotting test predictions for fold {fold+1}: {e}")
        
        # Compute average metrics for the fold
        avg_fold_dice = np.mean(fold_dice_scores) if fold_dice_scores else np.nan
        avg_fold_iou = np.mean(fold_iou_scores) if fold_iou_scores else np.nan
        avg_fold_sensitivity = np.mean(fold_sensitivity_scores) if fold_sensitivity_scores else np.nan
        avg_fold_specificity = np.mean(fold_specificity_scores) if fold_specificity_scores else np.nan
        avg_fold_f1 = np.mean(fold_f1_scores) if fold_f1_scores else np.nan
        avg_fold_auc = np.mean(fold_auc_scores) if fold_auc_scores else np.nan
        avg_fold_accuracy = np.mean(fold_accuracy_scores) if fold_accuracy_scores else np.nan  # Compute average accuracy
        print(f"Fold {fold+1} - Test Dice: {avg_fold_dice:.4f}, IoU: {avg_fold_iou:.4f}, Sensitivity: {avg_fold_sensitivity:.4f}, "
                    f"Specificity: {avg_fold_specificity:.4f}, F1: {avg_fold_f1:.4f}, AUC: {avg_fold_auc:.4f}, Accuracy: {avg_fold_accuracy:.4f}")
        
        # Append fold metrics to all scores
        all_dice_scores.append(fold_dice_scores)
        all_iou_scores.append(fold_iou_scores)
        all_sensitivity_scores.append(fold_sensitivity_scores)
        all_specificity_scores.append(fold_specificity_scores)
        all_f1_scores.append(fold_f1_scores)
        all_auc_scores.append(fold_auc_scores)
        all_accuracy_scores.append(fold_accuracy_scores)  # Append accuracy scores
    
    # Save averaged predicted masks
    for image_file, fold_probs_list in preds_dict.items():
        if fold_probs_list:
            avg_prob = np.mean(fold_probs_list, axis=0)
            pred_mask = (avg_prob > 0.5).astype(np.uint8) * 255
            mask_filename = f"{os.path.splitext(os.path.basename(image_file))[0]}_avg.png"
            mask_path = os.path.join(mask_output_dir, mask_filename)
            try:
                Image.fromarray(pred_mask).save(mask_path)
                print(f"Saved predicted mask: {mask_path}")
            except Exception as e:
                logger.error(f"Error saving predicted mask {mask_path}: {e}")
    
    # Compute average and standard deviation across folds for all metrics
    valid_dice = [np.mean(scores) for scores in all_dice_scores if scores and not np.isnan(scores).any()]
    valid_iou = [np.mean(scores) for scores in all_iou_scores if scores and not np.isnan(scores).any()]
    valid_sensitivity = [np.mean(scores) for scores in all_sensitivity_scores if scores and not np.isnan(scores).any()]
    valid_specificity = [np.mean(scores) for scores in all_specificity_scores if scores and not np.isnan(scores).any()]
    valid_f1 = [np.mean(scores) for scores in all_f1_scores if scores and not np.isnan(scores).any()]
    valid_auc = [np.mean(scores) for scores in all_auc_scores if scores and not np.isnan(scores).any()]
    valid_accuracy = [np.mean(scores) for scores in all_accuracy_scores if scores and not np.isnan(scores).any()]  # Compute valid accuracy
    
    avg_dice = np.mean(valid_dice) if valid_dice else np.nan
    avg_iou = np.mean(valid_iou) if valid_iou else np.nan
    avg_sensitivity = np.mean(valid_sensitivity) if valid_sensitivity else np.nan
    avg_specificity = np.mean(valid_specificity) if valid_specificity else np.nan
    avg_f1 = np.mean(valid_f1) if valid_f1 else np.nan
    avg_auc = np.mean(valid_auc) if valid_auc else np.nan
    avg_accuracy = np.mean(valid_accuracy) if valid_accuracy else np.nan  # Compute average accuracy
    
    std_dice = np.std(valid_dice) if valid_dice else np.nan
    std_iou = np.std(valid_iou) if valid_iou else np.nan
    std_sensitivity = np.std(valid_sensitivity) if valid_sensitivity else np.nan
    std_specificity = np.std(valid_specificity) if valid_specificity else np.nan
    std_f1 = np.std(valid_f1) if valid_f1 else np.nan
    std_auc = np.std(valid_auc) if valid_auc else np.nan
    std_accuracy = np.std(valid_accuracy) if valid_accuracy else np.nan  # Compute standard deviation for accuracy
    
    print(f"\nFinal Test Results (Averaged across {len(model_paths)} models):")
    print(f"Average Test Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    print(f"Average Test IoU: {avg_iou:.4f} ± {std_iou:.4f}")
    print(f"Average Test Sensitivity: {avg_sensitivity:.4f} ± {std_sensitivity:.4f}")
    print(f"Average Test Specificity: {avg_specificity:.4f} ± {std_specificity:.4f}")
    print(f"Average Test F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Average Test AUC: {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Average Test Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

    total_duration = time.time() - total_start_time
    print(f"Total execution time: {format_time(total_duration)}")
    
    return avg_dice, avg_iou, avg_sensitivity, avg_specificity, avg_f1, avg_auc, avg_accuracy, std_dice, std_iou, std_sensitivity, std_specificity, std_f1, std_auc, std_accuracy


model_paths = [f"./best_corrosion_model_fold{i+1}.pth" for i in range(3)]

avg_dice, avg_iou, avg_sensitivity, avg_specificity, avg_f1, avg_auc, avg_accuracy, std_dice, std_iou, std_sensitivity, \
std_specificity, std_f1, std_auc, std_accuracy = evaluate_test_set(dataset.test_pairs, model_paths, dataset.test_image_dir, \
                                                                   dataset.test_mask_dir, device) 