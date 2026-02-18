import os
import random
from torch.utils.data import DataLoader
from utils import EMDataset
from model import UNet
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cpu")

image_dir = "data/isbi-datasets-master/data/images"
labels_dir = "data/isbi-datasets-master/data/labels"

files = os.listdir(image_dir)
random.shuffle(files)

split = int(0.8 * len(files))
train_files = files[:split]
val_files = files[split:]

train_dataset = EMDataset(image_dir, labels_dir, train_files)
val_dataset = EMDataset(image_dir, labels_dir, val_files)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

def dice_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 9

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            val_dice += dice_score(probs, masks).item()

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Dice: {val_dice/len(val_loader):.4f}")
    print("-"*30)

torch.save(model.state_dict(), "model.pth")
