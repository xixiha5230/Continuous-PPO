import torch.nn as nn
import numpy as np

from torch.distributions import Categorical, Normal
from layers.StateNet import StateNetIR, StateNetI, weights_init_
from gymnasium import spaces


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_space, config: dict):
        super(ActorCritic, self).__init__()
        self.config = config

        self.conf_train = config['train']
        self.has_continuous_action = self.conf_train['has_continuous_action_space']
        self.device = self.conf_train['device']
        self.hidden_layer_size = self.conf_train['hidden_layer_size']

        self.conf_recurrence = config['recurrence']
        self.use_lstm = self.conf_recurrence['use_lstm']
        self.layer_type = self.conf_recurrence['layer_type']
        if self.use_lstm:
            self.hidden_state_size = self.conf_recurrence['hidden_state_size']
        if self.has_continuous_action:
            self.action_dim = action_space.shape[0]
            self.action_max = max(action_space.high)
        else:
            self.action_dim = action_space
        self.obs_space = obs_space
        if isinstance(obs_space, spaces.Tuple):
            self.state = StateNetIR(obs_space, 192)
            in_features_size = self.state.out_size
        elif len(obs_space.shape) == 1:
            in_features_size = obs_space.shape[0]
        elif len(obs_space.shape) == 3:
            self.state = StateNetI(obs_space, 128)
            in_features_size = self.state.out_size
        else:
            raise NotImplementedError(obs_space.shape)

        # gru
        if self.use_lstm:
            if self.layer_type == 'gru':
                self.rnn = nn.GRU(in_features_size,
                                  self.hidden_state_size, batch_first=True)
            elif self.layer_type == 'lsym':
                self.recurrent_layer = nn.LSTM(in_features_size, self.hidden_state_size, batch_first=True)
            else:
                raise NotImplementedError(self.layer_type)
            self.rnn.apply(weights_init_)

            after_rnn_size = self.hidden_state_size
        else:
            after_rnn_size = in_features_size

        # hidden layer
        self.lin_hidden = nn.Sequential(
            nn.Linear(after_rnn_size, self.hidden_layer_size),
            nn.ReLU()
        )
        self.lin_hidden.apply(weights_init_)

        # actor
        if self.has_continuous_action:
            ac_h = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
            nn.init.orthogonal_(ac_h.weight, np.sqrt(2))
            ac = nn.Linear(self.hidden_layer_size, self.action_dim)
            nn.init.orthogonal_(ac.weight, np.sqrt(0.01))
            self.mu = nn.Sequential()
            self.mu.append(ac_h).append(nn.ReLU())
            self.mu.append(ac).append(nn.Tanh())

            std = nn.Linear(self.hidden_layer_size, self.action_dim)
            nn.init.orthogonal_(std.weight, np.sqrt(0.01))
            self.sigma = nn.Sequential()
            self.sigma.append(std).append(nn.Softmax(dim=-1))
        else:
            ac_h = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
            nn.init.orthogonal_(ac_h.weight, np.sqrt(2))
            ac = nn.Linear(self.hidden_layer_size, self.action_dim)
            nn.init.orthogonal_(ac.weight, np.sqrt(0.01))
            self.mu = nn.Sequential()
            self.mu.append(ac_h)
            self.mu.append(nn.ReLU())
            self.mu.append(ac)
        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer_size, 1)
        )
        self.critic.apply(weights_init_)

    def forward(self, state, hidden_in=None, sequence_length=1):
        # complex input or image
        if isinstance(self.obs_space, spaces.Tuple) or len(self.obs_space.shape) == 3:
            feature = self.state(state)
        else:
            feature = state

        # lstm
        if self.use_lstm:
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
        else:
            hidden_out = None

        # hiddden
        feature = self.lin_hidden(feature)
        # actor
        if self.has_continuous_action:
            action_mean = self.action_max * self.mu(feature)
            # log_std = self.sigma.expand_as(action_mean)
            # action_std = torch.exp(log_std)
            action_std = self.sigma(feature)
            dist = Normal(action_mean, action_std)
        else:
            action_probs = self.mu(feature)
            dist = Categorical(logits=action_probs)

        # critic
        value = self.critic(feature)
        value = value.squeeze(-1)
        return dist, value, hidden_out if hidden_out != None else None
