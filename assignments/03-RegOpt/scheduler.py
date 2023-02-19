from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import torch


class CustomLRScheduler(_LRScheduler):
    """
    custom lr
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        initial_lr=0.01,
        eta_min=0,
        T_max=2,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.T_max = T_max
        self.lr = initial_lr
        self.last_epoch = last_epoch

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        get learning rate
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:

        pi = torch.acos(torch.zeros(1)) * 2
        if self.last_epoch == -1:
            lr = self.lr
        else:
            lr = self.eta_min + 0.5 * abs(self.lr - self.eta_min) * (
                1 + torch.cos(pi * self.last_epoch / self.T_max)
            )
        return lr
