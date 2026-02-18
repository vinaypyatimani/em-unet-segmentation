import torch
import matplotlib.pyplot as plt
from model import UNet
from utils import EMDataset
import os
import random

device = torch.device("cpu")

image_dir = "data/isbi-datasets-master/data/images"
labels_dir = "data/isbi-datasets-master/data/labels"

files = os.listdir(image_dir)
random.shuffle(files)

split = int(0.8 * len(files))
val_files = files[split:]

val_dataset = EMDataset(image_dir, labels_dir, val_files)

model = UNet().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

for i in range(3):
    image, mask = val_dataset[i]
    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        output = torch.sigmoid(model(image_batch))

    prediction = (output > 0.5).float()

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(image.squeeze(), cmap="gray")

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(mask.squeeze(), cmap="gray")

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(prediction.squeeze(), cmap="gray")

    plt.show()
