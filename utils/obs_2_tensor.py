import numpy as np
import torch


def _obs_2_tensor(obs: list, device: str):
    """obs ndarray to tensor,one obs type to one tensor
    Args:
        obs {list} -- observation
        device {str} -- 'cuda' or 'cpu'
    Returns:
        {torch.Tensor} -- tensor of state
    """
    assert isinstance(obs, list)
    obs = [torch.FloatTensor(np.array(o)).to(device) for o in obs]
    return obs
