import cv2
import torch
from torchvision import transforms, models
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/defect_model.pth"))
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)
    classes = ["defect", "good"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = transform(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
            label = classes[pred.item()]

        cv2.putText(frame, f"Prediction: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Defect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
