import torch
from tqdm import tqdm
from src.callbacks.checkpoint import ModelCheckpoint
from src.callbacks.early_stopping import EarlyStopping
import config

def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False, position=1)
    
    for images, labels in pbar:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = running_loss / len(loader)
        acc = correct / total

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{acc*100:.2f}%",
            lr=optimizer.param_groups[0]["lr"]
        )

    
    return avg_loss
    
    
def validate_one_epoch(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Val", leave=False, position=1) 
    
    with torch.no_grad():

        for images, labels in pbar:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc*100:.2f}%"
            )
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_catdog_classifier(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device, 
        epochs=20
    ):

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    checkpoint = ModelCheckpoint(model, config.MODEL_PATH, verbose=True)
    early_stopping = EarlyStopping(patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=3
    )

    for epoch in tqdm(range(epochs), desc="Epochs", position=0):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(
            model, 
            train_loader,
            optimizer, 
            criterion, 
            device
        )
        
        val_loss, val_accuracy = validate_one_epoch(
            model, 
            val_loader,  
            criterion, 
            device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Epoch's statistics
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

        # Saving model if the current model is better that the previous one
        checkpoint(val_loss)

        # Using early stopping to train with optimal epochs
        if early_stopping(val_loss):
            print(f"Early Stopping on {epoch} epoch.")
            break

    return history
        


