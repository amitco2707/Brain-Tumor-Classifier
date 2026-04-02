
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """
    A custom PyTorch Dataset for loading brain tumor MRI images.

    PyTorch requires any dataset to implement two methods:
      - __len__:     returns the total number of samples
      - __getitem__: returns one sample (image + label) by index

    Once we follow this contract, PyTorch can shuffle, batch, and
    split our data automatically.
    """

    def __init__(self, data_dir, transform=None):
        """
        Scans the data folder and builds a list of (image_path, label) pairs.

        Args:
            data_dir:  path to the folder containing 'yes' and 'no' subfolders
            transform: optional image transforms to apply (resize, normalize, etc.)
        """
        self.transform = transform
        self.samples = []  # will hold (file_path, label) tuples

        # Map each subfolder name to a numeric label
        # 1 = tumor present, 0 = no tumor
        label_map = {"yes": 1, "no": 0}

        for class_name, label in label_map.items():
            class_dir = os.path.join(data_dir, class_name)

            for filename in os.listdir(class_dir):
                # Only include image files, skip anything else (e.g. .DS_Store)
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(class_dir, filename)
                    self.samples.append((full_path, label))

    def __len__(self):
        # PyTorch calls this to know how many total images we have
        return len(self.samples)

    def __getitem__(self, index):
        """
        Loads and returns one image + its label by index.

        PyTorch's DataLoader will call this repeatedly (in random order
        if shuffling is on) to build each batch during training.
        """
        image_path, label = self.samples[index]

        # Open the image using Pillow and convert to RGB
        # We use RGB even though MRI scans are grayscale, because
        # ResNet-18 (our model) expects 3-channel input
        image = Image.open(image_path).convert("RGB")

        # Apply transforms if provided (resize, normalize, convert to tensor)
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """
    Returns two sets of transforms: one for training, one for validation/testing.

    Why two different sets?
    - Training transforms include random flips and rotation (data augmentation)
    - Val/test transforms are deterministic — no randomness — so results are consistent
    """

    # --- Training transforms ---
    # We apply some random changes to artificially increase the variety of our data.
    # This is called DATA AUGMENTATION. With only 253 images, the model could easily
    # memorize them. Augmentation makes each image look slightly different every epoch,
    # forcing the model to learn general patterns instead.
    train_transform = transforms.Compose([
        # Resize every image to 224x224 pixels
        # Why 224? ResNet-18 was originally designed for this size.
        transforms.Resize((224, 224)),

        # Randomly flip the image horizontally (like a mirror)
        # A tumor can appear on either side of the brain, so this is medically valid
        transforms.RandomHorizontalFlip(),

        # Randomly rotate up to 10 degrees
        # MRI scans aren't always perfectly aligned — small rotations help
        transforms.RandomRotation(10),

        # Convert the PIL image to a PyTorch tensor
        # This changes the format from (Height x Width x Channels) to
        # (Channels x Height x Width) — which is what PyTorch expects
        # It also scales pixel values from 0-255 to 0.0-1.0 automatically
        transforms.ToTensor(),

        # Normalize using the mean and std that ResNet-18 was originally trained with
        # Why? ResNet learned patterns from ImageNet data that was normalized this way.
        # Using the same normalization means its pretrained weights still "make sense"
        # for our images — we're speaking the same language as the pretrained model
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --- Validation / Test transforms ---
    # No augmentation here — we want a stable, repeatable result
    # so our accuracy measurements are honest and consistent
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_test_transform


def get_data_loaders(data_dir, batch_size=16):
    """
    Splits the dataset into train/val/test sets and wraps each in a DataLoader.

    A DataLoader is PyTorch's way of feeding data to the model in batches.
    Instead of loading all 253 images at once (slow, memory-heavy), it loads
    small groups (batches) one at a time during training.

    Args:
        data_dir:   path to the folder containing 'yes' and 'no' subfolders
        batch_size: how many images to feed the model at once (default: 16)

    Returns:
        three DataLoaders: train, validation, test
        and a dictionary with the size of each split
    """
    train_transform, val_test_transform = get_transforms()

    # Load the full dataset once (with no transform — we'll assign transforms
    # per split below, because train and val/test need different ones)
    full_dataset = BrainTumorDataset(data_dir=data_dir, transform=None)
    total = len(full_dataset)

    # Calculate split sizes: 70% train, 15% val, 15% test
    # We use int() to ensure whole numbers (can't have half an image)
    train_size = int(0.70 * total)  # ~177 images
    val_size   = int(0.15 * total)  # ~37 images
    test_size  = total - train_size - val_size  # remainder (~39 images)

    # random_split shuffles and divides the dataset
    # We set a generator seed so the split is the same every time we run
    # Without this, we'd get different train/test images each run —
    # making it impossible to fairly compare results across runs
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Assign the correct transforms to each split
    # We do this by wrapping the subset's dataset and overriding its transform
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform   = val_test_transform
    test_subset.dataset.transform  = val_test_transform

    # Wrap each split in a DataLoader
    # shuffle=True for training so the model doesn't see images in the same
    # order every epoch — order memorization is a subtle form of overfitting
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False)

    split_sizes = {
        "train": train_size,
        "val":   val_size,
        "test":  test_size,
    }

    return train_loader, val_loader, test_loader, split_sizes
