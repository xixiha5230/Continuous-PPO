import numpy as np
import torch


def _obs_2_tensor(obs: list, device: str):
    '''obs ndarray to tensor,one obs type to one tensor
    Args:
        obs {list} -- observation
        device {str} -- 'cuda' or 'cpu'
    Returns:
        {torch.Tensor} -- tensor of state
    '''
    if isinstance(obs, list):
        if not isinstance(obs[0], list):
            obs = [obs]
        obs = list(map(lambda x: torch.FloatTensor(np.array(x)).to(device), zip(*obs)))
    elif isinstance(obs, np.ndarray):
        if len(obs.shape) == 1:
            obs = [obs]
        obs = torch.FloatTensor(np.array(obs)).to(device)
    return obs
