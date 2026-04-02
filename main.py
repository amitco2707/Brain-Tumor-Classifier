import torch
from data_loader import BrainTumorDataset, get_data_loaders
from model import build_model
from training import train_model
from evaluation import evaluate_model, plot_history
import config


def explore_dataset():
    dataset      = BrainTumorDataset(data_dir=config.DATA_DIR, transform=None)
    num_tumor    = sum(1 for _, label in dataset.samples if label == 1)
    num_no_tumor = sum(1 for _, label in dataset.samples if label == 0)

    print("=" * 40)
    print("Dataset Summary")
    print("=" * 40)
    print(f"Total images   : {len(dataset)}")
    print(f"Tumor (yes)    : {num_tumor}")
    print(f"No tumor (no)  : {num_no_tumor}")
    print("=" * 40)


def run_training():
    # Build data loaders
    train_loader, val_loader, test_loader, split_sizes = get_data_loaders(config.DATA_DIR)
    print(f"\nSplit  ->  Train: {split_sizes['train']}  "
          f"Val: {split_sizes['val']}  Test: {split_sizes['test']}\n")

    # Build model
    model = build_model(num_classes=config.NUM_CLASSES)

    # Train — hyperparameters come from config.py so you only change them in one place
    model, history = train_model(model, train_loader, val_loader,
                                 num_epochs=config.NUM_EPOCHS,
                                 learning_rate=config.LEARNING_RATE)

    # Save the trained model weights to disk
    import os
    model_path = os.path.join(config.OUTPUTS_DIR, "brain_tumor_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Plot and save loss + accuracy curves (plot_history lives in evaluation/)
    plot_history(history, config.OUTPUTS_DIR)

    return model, test_loader


if __name__ == "__main__":
    explore_dataset()
    model, test_loader = run_training()
    evaluate_model(model, test_loader, config.OUTPUTS_DIR)
