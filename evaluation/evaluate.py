import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(model, test_loader, outputs_dir):
    """
    Runs the trained model on the test set and reports performance metrics.

    Args:
        model:       the trained model
        test_loader: DataLoader for the held-out test images
        outputs_dir: folder path to save the confusion matrix image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # disable training-specific behaviour (e.g. dropout)

    all_preds  = []  # every predicted label
    all_labels = []  # every true label

    with torch.no_grad():  # no gradient calculation needed — we're just predicting
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            # argmax picks the class with the highest score (0 or 1)
            preds = outputs.argmax(dim=1)

            # Move tensors to CPU and convert to plain Python lists
            # sklearn expects regular Python lists, not PyTorch tensors
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # ── CLASSIFICATION REPORT ───────────────────────────────────────
    # Prints precision, recall, F1 and support for each class
    print("=" * 50)
    print("Evaluation Report (Test Set)")
    print("=" * 50)
    print(classification_report(
        all_labels, all_preds,
        target_names=["No Tumor", "Tumor"]
    ))

    # ── CONFUSION MATRIX ────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)

    # ConfusionMatrixDisplay is sklearn's built-in visualiser for this
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Tumor", "Tumor"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")

    save_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_history(history, outputs_dir):
    """
    Saves a side-by-side plot of training/validation loss and accuracy.

    Moved here from main.py because plotting training results is an
    evaluation/visualization concern, not something the entry point should define.

    Args:
        history:     dict with keys train_loss, val_loss, train_acc, val_acc
                     each holding a list of values — one per epoch
        outputs_dir: folder path to save the plot image
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], label="Train acc")
    ax2.plot(epochs, history["val_acc"],   label="Val acc")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(outputs_dir, "training_curves.png")
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    plt.show()
