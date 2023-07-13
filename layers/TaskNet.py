import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils.weights_init import weights_init_


class TaskNet(nn.Module):
    """task encoding module"""

    def __init__(self, input_size: int, output_size: int):
        """
        Args:
            input_size {int} -- input task one_hot dimension
            output_size {int} -- output dimension
        """
        super(TaskNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )
        self.fc.apply(weights_init_)
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        """
        Args:
            x {torch.Tensor} -- tensor of sensor data shape like (batch, 400)
        """
        x = self.fc(x)
        return x


class TaskPredictNet(nn.Module):
    """
    Args:
        input_size {int} -- input task one_hot dimension
        hidden_size {int} -- hidden layer size
        output_size {int} -- output dimension
    """

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(TaskPredictNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        out = self.net(x)
        return out
