import torch
from torchvision import transforms
from PIL import Image
from src.model_utils import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(device)
model = model.to(device)
model.eval()

classes = ["Defective", "Non-Defective"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]
