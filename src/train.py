from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

def train_catdog_classifier(model, train_loader: DataLoader, criterion, optimizer, device, epochs: int=5):

    losses = []
    accuracies = []

    for epoch in range(epochs):

        model.train()
        
        loop = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs}")
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in loop:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculating accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            loop.set_postfix(loss=loss.item(), acc=f"{accuracy:.2f}%")

        # Epoch's statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print(f"\nResults of the {epoch+1} epoch:")
        print(f"Avg loss: {epoch_loss:.4f}")
        print(f"Accuracy: {epoch_acc:.2f}%")

    return losses, accuracies

def train_one_epoch():
    pass


