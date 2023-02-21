from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 14
    initial_learning_rate = 0.005
    initial_weight_decay = 0.000

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "T_0": 800,
        "eta_min": 3e-6,
        "T_mult": 9,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
