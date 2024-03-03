import torch
from torch import nn
import matplotlib.pyplot as plt
import torchinfo
from torchinfo import summary
from tqdm.auto import tqdm

print(torch.__version__)

# Setup device-agnostic code. If no GPU detected defaults to cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
device

import requests
import zipfile
from pathlib import Path

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

# Use the OS module to dig through each dir and explore its contents

import os
def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)

# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir, image_path

# Visualize an image and use the (PIL) python image library to do so
import random
from PIL import Image

# Set seed
random.seed(42)

# Get all the image paths using pathlib.Path.glob() function to find all files ending in .jpg
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Get a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# Get image class from path name
image_class = random_image_path.parent.stem
print(image_class)

# Open image
img = Image.open(random_image_path)

print("image height ", img.height, "image width ", img.width)
img

# Before we can use our image data with PyTorch we need to:

# Turn it into tensors (numerical representations of our images).
# Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader, we'll call these Dataset and DataLoader for short.

# Since we're working with a vision problem, we'll be looking at torchvision.datasets for our data loading functions as well as torchvision.transforms for preparing our data.

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Lets transform our images into tensors so we can work with them
# Transforms can also include: flipping images (a form of data augmentation), resizing images and so on.
# We compose a series of transforms we want to perform using torchvision.transform.Compose()

# Machine learning is all about harnessing the power of randomness and research shows that random transforms
# (like transforms.RandAugment() and transforms.TrivialAugmentWide()) generally perform better than hand-picked transforms.

#The main parameter to pay attention to in transforms.TrivialAugmentWide() is num_magnitude_bins=31.
#It defines how much of a range an intensity value will be picked to apply a certain transform, 0 being no range and 31 being maximum range (highest chance for highest intensity).

data_transform = transforms.Compose([
   # Resize image to 64x64
   transforms.Resize(size=(64,64)),
   # Flip the image randomly on the horizontal, where p is the probability of a flip, 0.5 =50% chance
   #transforms.RandomHorizontalFlip(p=0.5),
   transforms.TrivialAugmentWide(num_magnitude_bins=31),
   # turn the PIL image into a torch.Tensor
   transforms.ToTensor()

])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Here we will plot the images before applying the transforms and after to show the difference.
# This function take in the image paths (list of all images), the Transform.Compose,
# n is the number of images to pick and seed is for the random generator


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3)

# Lets create datasets from our train and test images by transforming and loading them into pytorch
# OPTION_1 We will use the torchvision.datasets.ImageFolder class.

# Use ImageFolder to create dataset(s)
from torchvision import datasets
train_data_augmented = datasets.ImageFolder(
    root=train_dir,
    transform=data_transform,
    target_transform=None #transform to perform on labales (if necessary)
)

test_data_simple = datasets.ImageFolder(
    root=test_dir,
    transform=test_transform)

print(f"Train data:\n{train_data_augmented}\nTest data:\n{test_data_simple}")


# Get classes as a list
class_names = train_data_augmented.classes
print(class_names)

# Get classes as a dict
class_dict = train_data_augmented.class_to_idx
print(class_dict)

# check the lenghts of each dataset
print(len(train_data_augmented),len(test_data_simple))

#Use indexing to find samples and their target data
img, label = train_data_augmented[0][0], train_data_augmented[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")


# Pytorch prefers CHW(color channels, height, width) however matplotlib prefers HWC(height, width, color channels)

# Rearrange the order of dimensions
img_permute = img.permute(1, 2, 0)

# Print out different shapes (before and after permute)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

# Plot the image
plt.figure(figsize=(10, 7))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(class_names[label], fontsize=14);


# We got our datasets but now we need to turn them into an iterable so
# the model can go through each one and learn the relationships betweensamples and targets (features and labels)

#Turn the train and test datasets into DataLoaders

from torch.utils.data import DataLoader
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)

# Turn datasets into iterables
# Set number of workers equal to your CPU count is a normal practice.
# Find your CPU count by using os.cpu_count()
train_dataloader = DataLoader(train_data_augmented,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)

# Do not usually have to shuffle test data
test_dataloader = DataLoader(test_data_simple,
                             batch_size=BATCH_SIZE,
                             shuffle=False, #dont shuffle test data because you want consistent test performance
                             num_workers=NUM_WORKERS)

train_dataloader, test_dataloader
next(iter(train_dataloader))[0].size(), next(iter(train_dataloader))[1].size()


# PyTorch training loop steps
#Create a CNN model

# The conv layers don't change the width/height of the features if you've set
# padding equal to (kernel_size - 1) / 2. Max pooling with kernel_size = stride = 2
# will decrease the width/height by a factor of 2 (rounded down if input shape is not even).
# To find the in_features you take the height/width as long as the are the same (e.g 64x64)
# and divide it by the kernel size to the power of the number of max pooling layers before the fully connected layer
# Example from below image input is 64x64 and 2 maxpooling layers of kernel size == 2
# gives 64/2*2 = 16x16 image which is multiply by the number of features from the last CN layer outchannel
# https://stackoverflow.com/questions/71385657/calculating-dimensions-of-fully-connected-layer

class SimpleCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.fc = nn.Linear(in_features=64*8*8, out_features=3)

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = x.reshape(x.size(0), -1) #flatten to a 1D tensor to be fed into the fc layer
    x = self.fc(x)
    return x


model = SimpleCNN().to(device)
model

# Use torchinfo to get an idea of the shapes going through our model

summary(model, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size


#Training step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

#Testing phase
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

#single call to combine Train and Test steps:
#Single call to combine Train and Test steps




# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# Code to start the model traiing and testing phases:
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 40

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Train model_1
model_1_results = train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# Check the model_0_results keys
model_1_results.keys()

#Code to plot the lose curve
from typing import Dict, List

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

plot_loss_curves(model_1_results)