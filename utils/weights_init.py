import numpy as np
import torch.nn as nn


def weights_init_(m: nn.Module):
    ''' orthogonal_ init the Linear and GRU module weight with sqrt(2)
        orthogonal_ init the GRU module bias with 0

    Args:
        shape {nn.Module} -- module
    '''
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.orthogonal_(m.weight, np.sqrt(2))
        m.bias.data.zero_()
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))
