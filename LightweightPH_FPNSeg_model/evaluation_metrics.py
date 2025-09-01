from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def sensitivity_score(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    target = target.float()
    true_positives = (pred * target).sum()
    actual_positives = target.sum()
    return (true_positives + 1e-6) / (actual_positives + 1e-6)

def specificity_score(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    target = target.float()
    true_negatives = ((1 - pred) * (1 - target)).sum()
    actual_negatives = (1 - target).sum()
    return (true_negatives + 1e-6) / (actual_negatives + 1e-6)

def f1_score(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    target = target.float()
    true_positives = (pred * target).sum()
    predicted_positives = pred.sum()
    actual_positives = target.sum()
    precision = (true_positives + 1e-6) / (predicted_positives + 1e-6)
    recall = (true_positives + 1e-6) / (actual_positives + 1e-6)
    return (2 * precision * recall + 1e-6) / (precision + recall + 1e-6)

def auc_score(pred_logits, target):
    pred_probs = torch.sigmoid(pred_logits).cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    try:
        return roc_auc_score(target_flat, pred_probs)
    except ValueError:
        print("AUC computation failed due to invalid input. Returning 0.0")
        return 0.0

def accuracy_score(pred_logits, gt_masks, threshold=0.5, epsilon=1e-6):
    """
    Calculate accuracy for binary segmentation.
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    binary_pred = (pred_logits > threshold).float()
    tp = (binary_pred * gt_masks).sum().item()
    tn = ((1 - binary_pred) * (1 -gt_masks)).sum().item()
    fp = (binary_pred * (1 - gt_masks)).sum().item()
    fn = ((1 - binary_pred) * gt_masks).sum().item()
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return accuracy
   
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7, l2_weight=1e-8, pos_weight=20.0, use_focal=True, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.l2_weight = l2_weight
        self.pos_weight = torch.tensor(pos_weight).to(device)
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_loss(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce
        return focal_loss.mean()

    def forward(self, pred, target, parameters=None):
        if self.use_focal:
            bce = self.focal_loss(pred, target)
        else:
            bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        dice = dice_loss(pred, target)
        l2 = sum(p.pow(2.0).sum() for p in parameters) * self.l2_weight if parameters else 0.0
        total_loss = self.bce_weight * bce + self.dice_weight * dice + l2
        return {
            'total_loss': total_loss,
            'bce_loss': bce,
            'dice_loss': dice,
            'l2_loss': l2
        }

