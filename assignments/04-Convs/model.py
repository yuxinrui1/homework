import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    A simple convolutional neural network for image classification on CIFAR-10 dataset.
        num_classes (int): The number of classes in the classification problem.


    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()

        # self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride =1,padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(16 * 8 * 8, 64)
        # self.fc2 = nn.Linear(32*32*32, num_classes)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: The output tensor with shape (batch_size, num_classes).
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1, 32 * 32 * 32)
        x = self.fc1(x)
        return x
