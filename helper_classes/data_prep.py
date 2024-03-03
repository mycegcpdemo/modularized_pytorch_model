import os
import random
import zipfile
from pathlib import Path
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


def create_directories_and_download_datasets():
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    # Unzip pizza, steak, sushi data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)


def create_datasets_and_dataloaders():
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Set seed
    random.seed(42)

    # Before we can use our image data with PyTorch we need to:
    # Turn it into tensors (numerical representations of our images).
    # Then create Datasets and DataLoaders.
    # The main parameter to pay attention to in transforms.TrivialAugmentWide() is num_magnitude_bins=31.
    # It defines how much of a range an intensity value will be picked to apply a certain transform, 0 being no range and 31 being maximum range (highest chance for highest intensity).


    data_transform = transforms.Compose([
        # Resize image to 64x64
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        # turn the PIL image into a torch.Tensor
        transforms.ToTensor()

    ])

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Let's create datasets from our train and test images by transforming and loading them into pytorch
    # Use ImageFolder to create dataset(s)
    train_data_augmented = datasets.ImageFolder(
        root=train_dir,
        transform=data_transform,
        target_transform=None  # transform to perform on labels (if necessary)
    )

    test_data_simple = datasets.ImageFolder(
        root=test_dir,
        transform=test_transform)

    # Get classes as a list
    class_names = train_data_augmented.classes

    # Get classes as a dict
    class_dict = train_data_augmented.class_to_idx

    # We got our datasets but now we need to turn them into an iterable so
    # the model can go through each one and learn the relationships between samples and targets (features and labels)
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    torch.manual_seed(42)

    # Turn datasets into iterables(DataLoaders)
    # Set number of workers equal to your CPU count is a normal practice.
    train_dataloader = DataLoader(train_data_augmented,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

    # Do not usually have to shuffle test data
    test_dataloader = DataLoader(test_data_simple,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,  # dont shuffle test data because you want consistent test performance
                                 num_workers=NUM_WORKERS)

    return train_dataloader, test_dataloader
