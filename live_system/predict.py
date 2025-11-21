import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import AccidentNet

# ------------------------------
# Load Model
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AccidentNet().to(device)
model.load_state_dict(torch.load("surakshan_model.pth", map_location=device))
model.eval()

# ------------------------------
# Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# ------------------------------
# Predict Function
# ------------------------------
def predict_frame(frame):
    """
    Takes a BGR OpenCV frame, converts to model input,
    sends it to the AccidentNet model, and returns:
        "ACCIDENT" or "NORMAL"
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        pred = torch.argmax(output, 1).item()

    return "ACCIDENT" if pred == 1 else "NORMAL"
