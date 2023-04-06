import torch
import torch.nn as nn
from layers.StateNet import weights_init_


class HiddenNet(nn.Module):
    ''' Hidden Module '''

    def __init__(self, in_size: int = 256, out_size: int = 256) -> None:
        '''
        Args:
            in_size {int} -- input feature dim
            out_size {int} -- output feature dim
        '''
        super(HiddenNet, self).__init__()
        self.lin_hidden = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU()
        )
        self.lin_hidden.apply(weights_init_)
        self.out_size = out_size

    def forward(self, feature: torch.Tensor):
        '''
        Args:
            feature {torch.Tensor} --- hight level feature
        '''
        feature = self.lin_hidden(feature)
        return feature
