import torch
import torch.nn as nn
from torch.distributions import Categorical
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
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )
        self.fc.apply(weights_init_)
        self.out_size = out_dim

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} -- tensor of sensor data shape like (batch, 400)
        '''
        x = self.fc(x)
        return x


class TaskPredictNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size) -> None:
        super(TaskPredictNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class VectorWithTask(nn.Module):
    def __init__(self, obs_space) -> None:
        super(VectorWithTask, self).__init__()
        assert len(obs_space) == 2
        feature_dim = obs_space[0].shape[0]
        task_dim = obs_space[1].shape[0]
        self.task_net = TaskNet(task_dim, 16)
        self.out_size = feature_dim + 16

    def forward(self, obs):
        assert len(obs) == 2
        feature = obs[0]
        task = self.task_net(obs[1])
        out = torch.cat((feature, task), dim=-1)
        return out


class ActorSelector(nn.Module):
    def __init__(self, in_size, actor_num) -> None:
        super(ActorSelector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, actor_num),
            nn.Softmax(dim=-1)
        )
        self.fc.apply(weights_init_)

    def forward(self, state):
        prob = self.fc(state)
        index = Categorical(probs=prob)
        return index.sample()
