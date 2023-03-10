import torch
import torch.nn as nn
# TODO Use tuple?
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces
from layers.RNN import RNN
from layers.Hidden import Hidden
from layers.Critic import Critic
from layers.Actor import GaussianActor
from layers.TaskNet import VectorWithTask
from layers.StateNet import StateNetUGV, StateNetImage, weights_init_


class ActorCritic(nn.Module):
    ''' Actor Critic Module '''

    def __init__(self, obs_space, action_space, config: dict):
        '''
        Args:
            obs_space {tuple} -- observation space
            action_space {tuple} -- action space
            config {dict} -- config dictionary
        '''
        super(ActorCritic, self).__init__()
        self.config = config
        self.conf_train = config['train']
        self.hidden_layer_size = self.conf_train['hidden_layer_size']
        self.conf_recurrence = config['recurrence']
        self.use_lstm = self.conf_recurrence['use_lstm']
        self.obs_space = obs_space

        # complex input
        if isinstance(obs_space, gym_spaces.Tuple) or isinstance(obs_space, gymnasium_spaces.Tuple):
            # UGV
            if(obs_space[0].shape == (84, 84, 3)):
                self.state = StateNetUGV(obs_space)
            # Simple Vector With Task ID
            elif len(obs_space[0].shape) == 1:
                self.state = VectorWithTask(obs_space)
            else:
                raise NotImplementedError(obs_space)
            in_features_size = self.state.out_size
        # simple vector
        elif len(obs_space.shape) == 1:
            in_features_size = obs_space.shape[0]
        # single image
        elif len(obs_space.shape) == 3:
            self.state = StateNetImage(obs_space)
            in_features_size = self.state.out_size
        else:
            raise NotImplementedError(obs_space.shape)

        # rnn
        if self.use_lstm:
            self.rnn = RNN(in_features_size, config)
            after_rnn_size = self.rnn.out_size
        else:
            after_rnn_size = in_features_size

        # hidden layer: out shape(self.hidden_layer_size,)
        self.lin_hidden = Hidden(after_rnn_size, self.hidden_layer_size)

        # actor: in shape(self.hidden_layer_size,)
        self.actor = GaussianActor(action_space, config)

        # critic: in shape(self.hidden_layer_size)
        self.critic = Critic(config=config)

    def forward(self, state, hidden_in: torch.Tensor = None, sequence_length: int = 1):
        '''
        Args:
            state {tensor, list} -- observation tensor
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
        Returns:
            {dist}: action dist
            {value}: value base on current state
            {hidden_out}: RNN hidden out feature  
        '''
        # complex input or image
        if isinstance(self.obs_space, gymnasium_spaces.Tuple) or isinstance(self.obs_space, gym_spaces.Tuple) or len(self.obs_space.shape) == 3:
            feature = self.state(state)
        else:
            feature = state

        # rnn
        if self.use_lstm:
            feature, hidden_out = self.rnn(feature, hidden_in, sequence_length)
        else:
            hidden_out = None

        # hiddden
        feature = self.lin_hidden(feature)

        # actor
        dist = self.actor(feature)

        # critic
        value = self.critic(feature)
        return dist, value, hidden_out if hidden_out != None else None
