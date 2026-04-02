
import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, val_loader, num_epochs=15, learning_rate=0.001):
    """
    Runs the full training loop: forward pass, loss calculation,
    backpropagation, and weight updates. Evaluates on validation set
    after every epoch.

    Args:
        model:         the neural network to train (ResNet-18 with our final layer)
        train_loader:  DataLoader for training images
        val_loader:    DataLoader for validation images
        num_epochs:    how many full passes through the training data (default: 15)
        learning_rate: how large each weight update step is (default: 0.001)

    Returns:
        model:    the trained model
        history:  dict of loss and accuracy lists, one value per epoch
    """

    # Use GPU if available, otherwise fall back to CPU
    # GPU training is ~10x faster, but CPU works fine for our small dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)  # move model weights to the chosen device

    # CrossEntropyLoss: standard loss function for classification
    # It measures how wrong the model's predictions are
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer: updates only the trainable parameters (our final layer)
    # It automatically adjusts step sizes per parameter — more reliable than SGD
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # We'll track these per epoch so we can plot them later
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   []
    }

    for epoch in range(num_epochs):

        # ── TRAINING PHASE ──────────────────────────────────────────
        model.train()  # tells the model we're training (enables dropout etc.)
        train_loss, train_correct = 0.0, 0

        # tqdm wraps the loader and shows a live progress bar
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()   # clear gradients from the previous batch
                                    # (PyTorch accumulates them by default)

            outputs = model(images) # forward pass: get predictions
            loss = criterion(outputs, labels)  # measure how wrong we were

            loss.backward()         # backprop: compute gradients
            optimizer.step()        # update weights using those gradients

            train_loss    += loss.item() * images.size(0)
            # argmax picks the class with the highest score
            train_correct += (outputs.argmax(1) == labels).sum().item()

        # ── VALIDATION PHASE ────────────────────────────────────────
        model.eval()  # tells the model we're evaluating (disables dropout etc.)
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():  # don't compute gradients during validation
                               # saves memory and speeds things up
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        # ── CALCULATE EPOCH AVERAGES ────────────────────────────────
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)

        epoch_train_loss = train_loss / n_train
        epoch_train_acc  = train_correct / n_train
        epoch_val_loss   = val_loss / n_val
        epoch_val_acc    = val_correct / n_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"  Train loss: {epoch_train_loss:.4f}  acc: {epoch_train_acc:.4f} | "
              f"Val loss: {epoch_val_loss:.4f}  acc: {epoch_val_acc:.4f}")

    print("\nTraining complete.")
    return model, history
