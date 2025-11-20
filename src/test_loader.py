from src.data_loader import create_dataloaders

train, val, test = create_dataloaders("data/DUTS")

for imgs, masks in train:
    print("Images:", imgs.shape)   # Expect [8, 3, 128, 128]
    print("Masks:", masks.shape)   # Expect [8, 1, 128, 128]
    break
