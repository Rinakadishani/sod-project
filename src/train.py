import torch
from torch import optim
from tqdm import tqdm

from src.data_loader import create_dataloaders
from src.model import CNN_SOD
from src.losses import BCEDiceLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(train_loader, desc="Training"):
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(imgs)
        loss, bce, iou = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss, _, _ = criterion(preds, masks)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    print("Using device:", DEVICE)

    train_loader, val_loader, test_loader = create_dataloaders(
    root_dir="data/DUTS",
    batch_size=8
    )
    model = CNN_SOD().to(DEVICE)

    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 20
    best_val = float("inf")
    patience = 3
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f},  Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), "final_model.pth")
            print("Saved best model!")
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")


if __name__ == "__main__":
    main()
