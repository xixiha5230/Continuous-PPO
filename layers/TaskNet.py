import torch
import torch.nn as nn
from layers.StateNet import weights_init_


class TaskNet(nn.Module):
    ''' task encoding module '''

    def __init__(self, in_dim: int, out_dim: int):
        '''
        Args:
            in_dim {int} -- input task one_hot dimension
            out_dim {int} -- output dimension
        '''
        super(TaskNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )
        self.fc.apply(weights_init_)

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} -- tensor of sensor data shape like (batch, 400)
        '''
        x = self.fc(x)
        return x


class VectorWithTask(nn.Module):
    def __init__(self, obs_space) -> None:
        super().__init__()
        assert len(obs_space) == 2
        feature_dim = obs_space[0].shape[0]
        task_dim = obs_space[1].shape[0]
        self.task_net = TaskNet(task_dim, 16)
        self.out_size = feature_dim + 16

    def forward(self, state):
        assert len(state) == 2
        feature = state[0]
        task = self.task_net(state[1])
        out = torch.cat((feature, task), dim=-1)
        return out
