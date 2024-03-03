import torch
from torch import nn
from torchinfo import summary


# Create a CNN model
class Model(nn.Module):

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

        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)  # flatten to a 1D tensor to be fed into the fc layer
        x = self.fc(x)
        return x


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SimpleCNN().to(device)
# # Use torchinfo to get an idea of the shapes going through our model
# summary(model, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size
