import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast
import matplotlib.pyplot as plt
import pandas as pd
import evaluation_metrics as mt
import dataset_load as dl
import sys
from sam2.build_sam import build_sam2


# Test dataset and loader
test_dataset = dl.CorrosionDataset(
    dl.test_pairs,
    image_dir=dl.test_image_dir,
    mask_dir=dl.test_mask_dir,
    target_size=(512, 512),
    transform=dl.test_transform,
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=mt.custom_collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save predicted masks
save_dir = "./sam_predicted_masks"
os.makedirs(save_dir, exist_ok=True)
amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam2_checkpoint = "./sam2_repo/checkpoints/sam2.1_hiera_large.pt"

# Evaluate for each fold and compute average metrics
num_folds = 3
all_fold_metrics = []
global_probs = []
global_targets = []
image_predictions = {}  

for fold in range(1, num_folds + 1):
    print(f"\n--- Evaluating Fold {fold} on Test Set ---")
    
    # Initialize model components
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", sam2_checkpoint, device=device)
    image_encoder = sam2_model.image_encoder.to(device)
    prompt_encoder = sam2_model.sam_prompt_encoder.to(device)
    mask_decoder = sam2_model.sam_mask_decoder.to(device)
    conv_256_to_32 = nn.Conv2d(256, 32, kernel_size=1).to(device)
    conv_128_to_64 = nn.Conv2d(256, 64, kernel_size=1).to(device)
    
    # Load best model for this fold
    best_model_path = f"./best_sam_based_model_fold{fold}.pth"
    if not os.path.exists(best_model_path):
        print(f"[ERROR] Model path not found for fold {fold}: {best_model_path}")
        continue
    checkpoint = torch.load(best_model_path, map_location=device)
    image_encoder.load_state_dict(checkpoint['image_encoder'])
    prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
    mask_decoder.load_state_dict(checkpoint['mask_decoder'])
    conv_256_to_32.load_state_dict(checkpoint['conv_256_to_32'])
    conv_128_to_64.load_state_dict(checkpoint['conv_128_to_64'])
    print(f"[INFO] Loaded best model for fold {fold}")
    
    image_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    
    fold_dice = []
    fold_iou = []
    fold_sensitivity = []
    fold_specificity = []
    fold_f1 = []
    fold_accuracy = [] 
    fold_probs = []
    fold_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Test Fold {fold}"):
            try:
                image = batch["image"].to(device, non_blocking=True)
                mask_gt = batch["mask"].to(device, non_blocking=True, dtype=torch.float32)
                points = batch["points"]
                labels = batch["labels"]
                image_file = batch["image_files"][0]  # since batch_size=1
                
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    # Encoder
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
                        point_coords = points[b].to(device).unsqueeze(0)
                        point_labels = labels[b].to(device).unsqueeze(0)
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

                    pred_prob = torch.sigmoid(mask_logits)

                # Compute per-image metrics
                dice, iou, sens, spec, f1, acc = mt.calculate_metrics(pred_prob, mask_gt)
                fold_dice.append(dice)
                fold_iou.append(iou)
                fold_sensitivity.append(sens)
                fold_specificity.append(spec)
                fold_f1.append(f1)
                fold_accuracy.append(acc)

                # Collect for global AUC
                fold_probs.append(pred_prob.flatten().cpu().numpy())
                fold_targets.append(mask_gt.flatten().cpu().numpy())

                # Store predictions for averaging
                if image_file not in image_predictions:
                    image_predictions[image_file] = []
                image_predictions[image_file].append(pred_prob.squeeze(0).cpu().numpy())

                # Save predicted mask
                pred_binary = (pred_prob > 0.5).squeeze(0).cpu().numpy().astype(np.uint8) * 255
                base_name = os.path.splitext(image_file)[0]
                pred_path = os.path.join(save_dir, f"fold{fold}_{base_name}.png")
                Image.fromarray(pred_binary).save(pred_path)
                print(f"[INFO] Saved predicted mask: {pred_path}")

            except Exception as e:
                print(f"[ERROR] Error in test batch: {e}")
                continue

    # Compute average metrics for this fold
    if fold_dice:
        avg_dice = np.mean(fold_dice)
        avg_iou = np.mean(fold_iou)
        avg_sensitivity = np.mean(fold_sensitivity)
        avg_specificity = np.mean(fold_specificity)
        avg_f1 = np.mean(fold_f1)
        avg_accuracy = np.mean(fold_accuracy)
        fold_probs_flat = np.concatenate(fold_probs)
        fold_targets_flat = np.concatenate(fold_targets)
        if np.unique(fold_targets_flat).size > 1:
            auc = mt.roc_auc_score(fold_targets_flat, fold_probs_flat)
        else:
            auc = 0.0  
        print(f"\n[INFO] Fold {fold} Test Metrics:")
        print(f"Dice: {avg_dice:.4f}")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Sensitivity: {avg_sensitivity:.4f}")
        print(f"Specificity: {avg_specificity:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"F1: {avg_f1:.4f}")
        print(f"Accuracy: {avg_accuracy:.4f}")
        
        all_fold_metrics.append({
            'dice': avg_dice,
            'iou': avg_iou,
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity,
            'auc': auc,
            'f1': avg_f1,
            'accuracy': avg_accuracy
        })
        
        global_probs.append(fold_probs_flat)
        global_targets.append(fold_targets_flat)
    else:
        print(f"[WARNING] No valid test batches for fold {fold}")

# Save averaged predicted masks
print("\n[INFO] Saving averaged predicted masks")
for image_file, pred_probs_list in image_predictions.items():
    try:
        # Average predictions across folds
        avg_pred_prob = np.mean(pred_probs_list, axis=0)
        avg_pred_binary = (avg_pred_prob > 0.5).astype(np.uint8) * 255
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        pred_path = os.path.join(save_dir, f"{base_name}_avg_pred.png")
        Image.fromarray(avg_pred_binary).save(pred_path)
        print(f"[INFO] Saved averaged predicted mask: {pred_path}")
    except Exception as e:
        print(f"[ERROR] Error saving averaged predicted mask for {image_file}: {e}")

# Compute overall average across folds
if all_fold_metrics:
    avg_metrics = pd.DataFrame(all_fold_metrics).mean()
    global_probs_all = np.concatenate(global_probs)
    global_targets_all = np.concatenate(global_targets)
    if np.unique(global_targets_all).size > 1:
        overall_auc = mt.roc_auc_score(global_targets_all, global_probs_all)
    else:
        overall_auc = 0.0
    print("\n[INFO] Overall Average Test Metrics Across Folds:")
    print(f"Dice: {avg_metrics['dice']:.4f}")
    print(f"IoU: {avg_metrics['iou']:.4f}")
    print(f"Sensitivity: {avg_metrics['sensitivity']:.4f}")
    print(f"Specificity: {avg_metrics['specificity']:.4f}")
    print(f"AUC: {overall_auc:.4f}")
    print(f"F1: {avg_metrics['f1']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
else:
    print("[ERROR] No metrics computed across folds")



