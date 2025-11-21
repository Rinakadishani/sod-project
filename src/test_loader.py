from src.data_loader import create_dataloaders

train, val, test = create_dataloaders("data/DUTS")

for imgs, masks in train:
    print("Images:", imgs.shape)   
    print("Masks:", masks.shape)  
    break
