from typing import List

from torch.optim.lr_scheduler import _LRScheduler
import math
import torch


class CustomLRScheduler(_LRScheduler):
    """
    custom lr
    """

    def __init__(self, optimizer, T_0, T_mult, eta_min, last_epoch=-1, verbose=False):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        get learning rate
        """

        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
