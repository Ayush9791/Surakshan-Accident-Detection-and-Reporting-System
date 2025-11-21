import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import AccidentNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------
# DATASET
# -----------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------
# MODEL + TRAINING
# -----------------------
model = AccidentNet().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10  # Increase for better accuracy

print("Starting training...\n")

for epoch in range(epochs):
    total_loss = 0
    correct = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()

    accuracy = correct / len(dataset) * 100

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

# -----------------------
# SAVE MODEL
# -----------------------
torch.save(model.state_dict(), "surakshan_model.pth")
print("\n✅ Model saved as surakshan_model.pth")
