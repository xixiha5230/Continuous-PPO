import torch
import torch.nn as nn
from typing import Union, Tuple
from gymnasium import spaces
import numpy as np
from math import floor


def weights_init_(m: nn.Module):
    ''' orthogonal_ init the Linear and GRU module weight with sqrt(2)
        orthogonal_ init the GRU module bias with 0

    Args:
        shape {nn.Module} -- module
    '''
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, Conv1d):
        nn.init.orthogonal_(m.weight, np.sqrt(2))
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))


def conv1d_output_size(length: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1):
    ''' calculate conv1d output size
        comv1d input shape is like (chanel, length)

    Args:
        length {int} -- length of feature
        kernel_size {int} -- conv1d kernel size
        stride {int} -- conv1d stride length
        padding {int} -- conv1d padding size
        dilation {int} -- conv1d dilation size
    '''
    l_out = floor(((length + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1)
    return l_out


def conv2d_output_shape(
    h_w: Tuple[int, int],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[int, int]:
    '''Calculates the output shape (height and width) of the output of a convolution layer.
        kernel_size, stride, padding and dilation correspond to the inputs of the 
        torch.nn.Conv2d layer (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

    Args:
        h_w {Tuple[int, int]} -- The height and width of the input
        kernel_size {Union[int, Tuple[int, int]]} -- The size of the kernel of the convolution
        stride {int} -- The stride of the convolution
        padding {int} -- The padding of the convolution
        dilation {int} -- The dilation of the convolution
    '''
    if not isinstance(kernel_size, tuple):
        kernel_size = (int(kernel_size), int(kernel_size))
    h = floor(((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class AtariImage(nn.Module):
    ''' image convolution module like Open AI Atari '''

    def __init__(self, shape):
        '''
        Args:
            shape {tuple} -- The shape of image like (84, 84, 3)
        '''
        super(AtariImage, self).__init__()
        conv_1_hw = conv2d_output_shape((shape[0], shape[1]), 8, 4)
        conv_2_hw = conv2d_output_shape(conv_1_hw, 4, 2)
        conv_3_hw = conv2d_output_shape(conv_2_hw, 3, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(shape[2], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
        )
        self.conv.apply(weights_init_)
        self.out_size = conv_3_hw[0] * conv_3_hw[1] * 64

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} -- image tensor shape like (84, 84, 3)
        '''
        # TODO 环境采样时使用（3，84，84）不用每次都permute
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.out_size)
        return x


class Conv1d(nn.Module):
    ''' dense sensor data convolution module '''

    def __init__(self, length: int, channel: int, out_size: int):
        '''
        Args:
            length {int} -- data length
            channel {int} -- data chanel
            out_size {int} -- output dimension
        '''
        super(Conv1d, self).__init__()

        conv_1_l = conv1d_output_size(length, 8, 4)
        conv_2_l = conv1d_output_size(conv_1_l, 4, 2)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, 16, 8, 4),
            nn.ReLU(),
            nn.Conv1d(16, 32, 4, 2),
            nn.ReLU(),
        )
        self.conv.apply(weights_init_)

        self.fc_input = conv_2_l * 32
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input, out_size),
            nn.ReLU()
        )
        self.fc.apply(weights_init_)
        self.out_size = out_size

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} -- tensor of sensor data shape like (batch, 400)
        '''
        batch = x.shape[-2]
        x = x.reshape(x.shape[-2], 2, x.shape[-1] // 2)
        hidden = self.conv(x)
        hidden = hidden.reshape(batch, self.fc_input)
        x = self.fc(hidden)
        return x


class UGVImage(nn.Module):
    ''' UGV image data convolution module '''

    def __init__(self, shape, out_size) -> None:
        '''
        Args:
            shape {int} -- image shape
            out_size {int} -- output dimension
        '''
        super(UGVImage, self).__init__()
        self.conv2d = AtariImage(shape)
        self.fc_input = self.conv2d.out_size
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input, out_size),
            nn.ReLU()
        )
        self.fc.apply(weights_init_)
        self.out_size = out_size

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x {torch.Tensor} -- tensor of sensor data shape like (batch, 84,84,3)
        '''
        x = self.conv2d(x)
        x = self.fc(x)
        return x


class StateNetUGV(nn.Module):
    ''' UGV environment data process module '''

    def __init__(self, obs_space: spaces.Tuple) -> None:
        '''
        Args:
            obs_space {spaces.Tuple} -- shape is ((84, 84, 3), (400, ))
        '''
        assert obs_space[0].shape == (84, 84, 3)
        assert obs_space[1].shape == (400,)
        super(StateNetUGV, self).__init__()

        self.conv1d = Conv1d(obs_space[1].shape[0] // 2, 2, 64)
        self.conv2d = UGVImage(obs_space[0].shape, 64)
        self.out_size = self.conv1d.out_size + self.conv2d.out_size

    def forward(self, state: list):
        '''
        Args:
            state {list} -- state is list of [image tensor, sensor tensor]
        '''
        img_batch = state[0]
        ray_batch = state[1]

        img = self.conv2d(img_batch)
        ray = self.conv1d(ray_batch)

        x = torch.cat([img, ray], dim=-1)
        return x


class StateNetImage(nn.Module):
    ''' single image process module '''

    def __init__(self, obs_space: spaces.Tuple) -> None:
        '''
        Args:
            obs_space {spaces.Tuple} -- shape is (84, 84, 3)
        '''
        assert obs_space.shape == (84, 84, 3)
        super(StateNetImage, self).__init__()
        self.conv2d = AtariImage(obs_space.shape)
        self.out_size = self.conv2d.out_size

    def forward(self, state: torch.Tensor):
        '''
        Args:
            state {torch.Tensor} -- tensor of image 
        '''
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        x = self.conv2d(state)
        return x
