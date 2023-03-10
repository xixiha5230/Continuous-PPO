import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from layers.StateNet import weights_init_


class RNN(nn.Module):
    ''' RNN Module '''

    def __init__(self, in_size: int, config: dict,) -> None:
        '''
        Args:
            in_size {int} -- input feature size
            config {dict} -- config dictionary
        '''
        super(RNN, self).__init__()
        self.config = config
        self.conf_recurrence = config['recurrence']
        self.use_lstm = self.conf_recurrence['use_lstm']
        self.layer_type = self.conf_recurrence['layer_type']
        self.hidden_state_size = self.conf_recurrence['hidden_state_size']
        if self.layer_type == 'gru':
            self.rnn = nn.GRU(in_size,
                              self.hidden_state_size, batch_first=True)
        elif self.layer_type == 'lstm':
            self.rnn = nn.LSTM(in_size, self.hidden_state_size, batch_first=True)
        else:
            raise NotImplementedError(self.layer_type)
        self.rnn.apply(weights_init_)
        self.out_size = self.hidden_state_size

    def forward(self, feature: torch.Tensor, hidden_in: torch.Tensor, sequence_length: int = 1):
        '''
        Args:
            feature {torch.Tensor} -- feature
            hidden_in {torch.Tensor} -- hidden in feature
            sequence_length {int} -- rnn sequence length
        '''
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            feature, hidden_out = self.rnn(feature.unsqueeze(1), hidden_in)
            # Remove sequence length dimension
            feature = feature.squeeze(1)
        else:
            feature_shape = tuple(feature.size())
            feature = feature.reshape((feature_shape[0] // sequence_length), sequence_length, feature_shape[1])
            feature, hidden_out = self.rnn(feature, hidden_in)
            out_shape = tuple(feature.size())
            feature = feature.reshape(out_shape[0] * out_shape[1], out_shape[2])
        return feature, hidden_out
