import torch
import torch.nn as nn

def iou_score(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()

    intersection = (pred_bin * target).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3]) - intersection

    return ((intersection + eps) / (union + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        iou = iou_score(pred, target)

        total_loss = bce_loss + 0.5 * (1 - iou)

        return total_loss, bce_loss.item(), iou.item()
