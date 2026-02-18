import torch
from sklearn.metrics import precision_score, recall_score
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset import get_transforms
from train import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_dataset = datasets.ImageFolder("data/val", transform=get_transforms(False))
val_loader = DataLoader(val_dataset, batch_size=32)

model = CNN().to(device)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()

        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)

print("Precision:", precision)
print("Recall:", recall)
