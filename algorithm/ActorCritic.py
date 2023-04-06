import torch
import torch.nn as nn
# TODO Use tuple?
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces
from layers.RNN import RNN
from layers.Hidden import HiddenNet
from layers.Critic import Critic, MultiCritic
from layers.Actor import GaussianActor, MultiGaussianActor
from layers.TaskNet import VectorWithTask, ActorSelector, TaskNet
from layers.StateNet import ObsNetUGV, ObsNetImage


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
        self.multi_task = self.conf_train['multi_task']
        self.conf_recurrence = config['recurrence']
        self.use_lstm = self.conf_recurrence['use_lstm']
        self.obs_space = obs_space

        # complex input
        if isinstance(obs_space, gym_spaces.Tuple) or isinstance(obs_space, gymnasium_spaces.Tuple):
            # UGV
            if(obs_space[0].shape == (84, 84, 3)):
                self.obs_net = ObsNetUGV(obs_space)
                in_features_size = self.obs_net.out_size
            # Simple Vector With Task ID shape like((17,), (4,))
            elif self.multi_task and len(obs_space[0].shape) == 1 and len(obs_space[1].shape) == 1:
                self.task_num = len(self.config.get('task', []))
                in_features_size = self.obs_space[0].shape[0]
            else:
                raise NotImplementedError(obs_space)
        # simple vector
        elif len(obs_space.shape) == 1:
            in_features_size = obs_space.shape[0]
        # single image
        elif len(obs_space.shape) == 3:
            self.obs_net = ObsNetImage(obs_space)
            in_features_size = self.obs_net.out_size
        else:
            raise NotImplementedError(obs_space.shape)

        # rnn
        if self.use_lstm:
            self.rnn_net = RNN(in_features_size, config)
            after_rnn_size = self.rnn_net.out_size
        else:
            after_rnn_size = in_features_size

        # hidden layer: out shape(self.hidden_layer_size,)
        self.hidden_net = HiddenNet(after_rnn_size, self.hidden_layer_size)

        # actor: in shape(self.hidden_layer_size,)
        if self.multi_task:
            self.actor = MultiGaussianActor(config, self.hidden_layer_size, action_space, self.task_num)
            self.critic = MultiCritic(self.hidden_layer_size, 1, config=config, task_num=self.task_num)
        else:
            self.actor = GaussianActor(config, self.hidden_layer_size, action_space)
            self.critic = Critic(self.hidden_layer_size, 1, config=config)

    def forward(self, obs, hidden_in: torch.Tensor = None, sequence_length: int = 1, module_index: int = -1):
        '''
        Args:
            state {tensor, list} -- observation tensor
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {dist}: action dist
            {value}: value base on current state
            {hidden_out}: RNN hidden out feature  
        '''
        # complex input or image or multi_task(obs and task id)
        if isinstance(self.obs_space, (gymnasium_spaces.Tuple, gym_spaces.Tuple)) or len(self.obs_space.shape) == 3:
            if self.multi_task:
                feature = self.obs_net(obs[0]) if isinstance(obs[0], list) else obs[0]
            else:
                feature = self.obs_net(obs)
        else:
            feature = obs

        # rnn
        if self.use_lstm:
            feature, hidden_out = self.rnn_net(feature, hidden_in, sequence_length)
        else:
            hidden_out = None

        # hiddden
        feature = self.hidden_net(feature)

        if self.multi_task:
            # select actor
            dist = self.actor(feature, module_index)
            # critic
            value = self.critic(feature, module_index)
        else:
            # actor
            dist = self.actor(feature)
            # critic
            value = self.critic(feature)
        return dist, value, hidden_out if hidden_out != None else None
