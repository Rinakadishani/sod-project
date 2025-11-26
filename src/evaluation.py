import os
import torch
import matplotlib.pyplot as plt

from src.data_loader import create_dataloaders
from src.model import CNN_SOD
from src.losses import iou_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dice_score(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()
    return (2 * inter + eps) / (total + eps)

def precision_score(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    TP = (pred_bin * target).sum()
    FP = (pred_bin * (1 - target)).sum()
    return (TP + eps) / (TP + FP + eps)

def recall_score(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    TP = (pred_bin * target).sum()
    FN = ((1 - pred_bin) * target).sum()
    return (TP + eps) / (TP + FN + eps)

def f1_score(pred, target, eps=1e-6):
    P = precision_score(pred, target, eps)
    R = recall_score(pred, target, eps)
    return (2 * P * R) / (P + R + eps)

def evaluate_model(model, test_loader):
    model.eval()
    ious, dices = [], []
    precisions, recalls, f1scores = [], [], []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)

            ious.append(iou_score(preds, masks).item())
            dices.append(dice_score(preds, masks).item())
            precisions.append(precision_score(preds, masks).item())
            recalls.append(recall_score(preds, masks).item())
            f1scores.append(f1_score(preds, masks).item())

    return (
        sum(ious) / len(ious),
        sum(dices) / len(dices),
        sum(precisions) / len(precisions),
        sum(recalls) / len(recalls),
        sum(f1scores) / len(f1scores),
    )


def save_visuals(model, test_loader, out_dir="outputs/visuals"):
    os.makedirs(out_dir, exist_ok=True)

    model.eval()

    imgs, masks = next(iter(test_loader))
    imgs = imgs.to(DEVICE)
    masks = masks.to(DEVICE)

    with torch.no_grad():
        preds = model(imgs)

    for i in range(min(5, len(imgs))):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(imgs[i].cpu().permute(1, 2, 0))
        axes[0].set_title("Input image")

        axes[1].imshow(masks[i].cpu().squeeze(), cmap="gray")
        axes[1].set_title("Actual Mask")

        axes[2].imshow(preds[i].cpu().squeeze(), cmap="gray")
        axes[2].set_title("Predicted mask")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{i}.png")
        plt.close()


def main():
    print("Using device:", DEVICE)

    _, _, test_loader = create_dataloaders(
        root_dir="data/DUTS",
        batch_size=8
    )

    model = CNN_SOD().to(DEVICE)
    model.load_state_dict(torch.load("final_model.pth", map_location=DEVICE))

    iou, dice, precision, recall, f1 = evaluate_model(model, test_loader)

    print("\nFinal Evaluation Results:")
    print(f"IoU:       {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")


    save_visuals(model, test_loader)

    print("Saved prediction visuals to outputs/visuals/")


if __name__ == "__main__":
    main()