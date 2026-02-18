import torch
import torch.nn as nn
import torchvision.models as models
import os

MODEL_PATH = "models/model.pth"

def get_model(device):
    os.makedirs("models", exist_ok=True)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    if not os.path.exists(MODEL_PATH):
        torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model
