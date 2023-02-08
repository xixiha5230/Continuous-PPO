import torch
import torch.nn as nn
from typing import Union, Tuple
from gym import spaces


def weights_init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def conv1d_output_size(
    length: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
):
    from math import floor

    l_out = floor(
        ((length + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    )
    return l_out


def conv2d_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output shape (height and width) of the output of a convolution layer.
    kernel_size, stride, padding and dilation correspond to the inputs of the
    torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
    :param h_w: The height and width of the input.
    :param kernel_size: The size of the kernel of the convolution (can be an int or a
    tuple [width, height])
    :param stride: The stride of the convolution
    :param padding: The padding of the convolution
    :param dilation: The dilation of the convolution
    """
    from math import floor

    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(
        ((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


class Conv2d(nn.Module):
    def __init__(self, shape, hidden_dim, out_dim):
        super(Conv2d, self).__init__()
        conv_1_hw = conv2d_output_shape((shape[0], shape[1]), 8, 4)
        conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv2d_output_shape(conv_2_hw, 3, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(shape[2], 32, [8, 8], [4, 4]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, [4, 4], [2, 2]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        self.fc_w = 64 * conv_3_hw[0] * conv_3_hw[1]
        self.fc = nn.Sequential(
            nn.Linear(self.fc_w, out_dim),
            nn.ReLU(),
        )

    # shape(1,84,84,4)
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.fc_w)
        x = self.fc(x)
        return x


class Conv1d(nn.Module):
    def __init__(self, length, channel, out_dim):
        super(Conv1d, self).__init__()

        conv_1_l = conv1d_output_size(length, 8, 4)
        conv_2_l = conv1d_output_size(conv_1_l, 4, 2)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, 16, 8, 4),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, 4, 2),
            nn.LeakyReLU(),
        )
        self.fc_input = conv_2_l * 32
        self.fc = nn.Sequential(nn.Linear(self.fc_input, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor):
        batch = x.shape[-2]
        x = x.reshape(x.shape[-2], 2, x.shape[-1] // 2)
        hidden = self.conv(x)
        hidden = hidden.reshape(batch, self.fc_input)
        x = self.fc(hidden)
        return x


class StateNetIR(nn.Module):
    def __init__(self, obs_space: spaces.Tuple, out_dim: int) -> None:
        assert obs_space[0].shape == (84, 84, 3)
        assert obs_space[1].shape == (400,)
        super(StateNetIR, self).__init__()

        self.conv1d = Conv1d(
            obs_space[1].shape[0] // 2, 2, 16
        )
        self.conv2d = Conv2d(obs_space[0].shape, hidden_dim=256, out_dim=64)
        self.fc = nn.Sequential(
            nn.Linear(16 + 64, out_dim), nn.ReLU()
        )
        self.apply(weights_init_)

    def forward(self, state):
        img_batch = state[0]
        ray_batch = state[1]

        img = self.conv2d(img_batch)
        ray = self.conv1d(ray_batch)

        x = self.fc(torch.cat([img, ray], dim=-1))
        return x
