import torch
import torch.nn as nn

from utils.weights_init import weights_init_


class HiddenNet(nn.Module):
    """Hidden Module"""

    def __init__(self, input_size: int = 256, output_size: int = 256) -> None:
        """
        Args:
            input_size {int} -- input feature dim
            output_size {int} -- output feature dim
        """
        super(HiddenNet, self).__init__()
        self.lin_hidden = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
        self.lin_hidden.apply(weights_init_)
        self.output_size = output_size

    def forward(self, feature: torch.Tensor):
        """
        Args:
            feature {torch.Tensor} --- hight level feature
        """
        feature = self.lin_hidden(feature)
        return feature
