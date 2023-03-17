import torch
import torch.nn as nn

# import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A simple convolutional neural network for image classification on CIFAR-10 dataset.
        num_classes (int): The number of classes in the classification problem.


    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(8 * 8 * 8, num_classes)

    def forward(self, x) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 8 * 8 * 8)
        x = self.fc(x)
        return x
