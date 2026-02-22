import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model

def train_model(epochs=5, lr=0.001):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    train_loader, test_loader = get_dataloaders()

    model = get_model()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f} Accuracy: {accuracy:.2f}%")

    return model


if __name__ == "__main__":
    train_model()