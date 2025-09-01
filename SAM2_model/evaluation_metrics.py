from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def custom_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    points = [item["points"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_files = [item["image_file"] for item in batch]
    return {
        "image": images,
        "mask": masks,
        "points": points,
        "labels": labels,
        "image_files": image_files
    }


# Loss and metrics
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(1)
    return ((2. * intersection + 1e-6) / (pred_flat.sum(1) + target_flat.sum(1) + 1e-6)).mean().item()

def iou_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    intersection = (pred * target).sum((1, 2))
    union = (pred + target - pred * target).sum((1, 2))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5, pos_weight=20.0, smooth=1e-6, l2_lambda=0):
#         super().__init__()
#         self.bce_weight = bce_weight
#         self.pos_weight = pos_weight
#         self.smooth = smooth
#         self.l2_lambda = l2_lambda
#         self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='mean')

#     def forward(self, logits, targets, parameters=None):
#         targets = targets.float()
#         bce_loss = self.bce_loss(logits, targets)
#         probs = torch.sigmoid(logits)
#         intersection = (probs * targets).sum(dim=[1, 2])
#         union = probs.sum(dim=[1, 2]) + targets.sum(dim=[1, 2])
#         dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
#         dice_loss = dice_loss.mean()
#         loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
#         if parameters is not None and self.l2_lambda > 0:
#             l2_loss = sum(p.pow(2.0).sum() for p in parameters if p.requires_grad)
#             num_params = sum(p.numel() for p in parameters if p.requires_grad)
#             l2_loss = l2_loss / num_params if num_params > 0 else l2_loss
#             loss += self.l2_lambda * l2_loss
#         return {
#             'total_loss': loss,
#             'bce_loss': bce_loss,
#             'dice_loss': dice_loss,
#             'l2_loss': self.l2_lambda * l2_loss if parameters is not None and self.l2_lambda > 0 else torch.tensor(0.0)
#         }


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7, l2_weight=1e-8, pos_weight=20.0, smooth=1e-6, use_focal=True, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.l2_weight = float(l2_weight)
        self.smooth = float(smooth)
        self.use_focal = bool(use_focal)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

        # Register pos_weight buffer for device-safe handling
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

        # BCE loss module 
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='mean')

    def focal_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce
        return loss.mean()

    def forward(self, logits, targets, parameters=None):
        targets = targets.to(dtype=logits.dtype)

        # BCE or Focal loss
        if self.use_focal:
            bce_loss = self.focal_loss(logits, targets)
        else:
            bce_loss = self.bce_loss(logits, targets)

        # Dice loss calculation (binary)
        probs = torch.sigmoid(logits)
        N = probs.shape[0]
        probs_f = probs.reshape(N, -1)
        targets_f = targets.reshape(N, -1)

        intersection = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        # Combine BCE and Dice 
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        # L2 regularization normalized by number of params
        if parameters is not None and self.l2_weight > 0:
            params = [p for p in parameters if p.requires_grad]
            if len(params) > 0:
                l2_sum = sum(p.pow(2).sum() for p in params)
                num_params = sum(p.numel() for p in params)
                l2_norm = l2_sum / max(1, num_params)
            else:
                l2_norm = torch.zeros((), device=logits.device, dtype=logits.dtype)
            l2_loss = self.l2_weight * l2_norm
            loss = loss + l2_loss
        else:
            l2_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        return {
            'total_loss': loss,
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            'l2_loss': l2_loss
        }

    
# Metrics calculation function
def calculate_metrics(pred_prob, target, threshold=0.5, epsilon=1e-6):
    binary_pred = (pred_prob > threshold).float()
    tp = (binary_pred * target).sum().item()
    tn = ((1 - binary_pred) * (1 - target)).sum().item()
    fp = (binary_pred * (1 - target)).sum().item()
    fn = ((1 - binary_pred) * target).sum().item()
    
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = sensitivity
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    dice = 2 * tp / (2 * tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    
    return dice, iou, sensitivity, specificity, f1, accuracy

