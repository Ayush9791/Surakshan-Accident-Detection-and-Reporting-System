import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import AccidentNet

# ------------------------------
# LOAD MODEL
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AccidentNet().to(device)
model.load_state_dict(torch.load("surakshan_model.pth", map_location=device))
model.eval()

# ------------------------------
# TRANSFORM
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ------------------------------
# FRAME PREDICTION
# ------------------------------
def predict_frame(frame):
    # BGR → RGB → PIL
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)
        pred = torch.argmax(out, 1).item()

    return "ACCIDENT" if pred == 1 else "NORMAL"


# ------------------------------
# START CAMERA
# ------------------------------
cap = cv2.VideoCapture(0)  
# For RTSP CCTV:
# cap = cv2.VideoCapture("rtsp://user:pass@IP:port/stream")

print("Starting live accident detection... Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    result = predict_frame(frame)

    # Red if accident, Green if normal
    color = (0, 0, 255) if result == "ACCIDENT" else (0, 255, 0)

    # Put text
    cv2.putText(frame, result, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Surakshan - Basic Live Detection", frame)

    # ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
