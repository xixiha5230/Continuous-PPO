import torch
import torch.nn as nn
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces
from layers.RNN import RNN
from algorithm.RND import RND
from layers.Critic import Critic
from layers.Hidden import HiddenNet
from layers.TaskNet import TaskNet, TaskPredictNet
from layers.StateNet import ObsNetUGV, ObsNetImage
from layers.Actor import GaussianActor, MultiGaussianActor


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
        conf_train = config['train']
        self.hidden_layer_size = conf_train['hidden_layer_size']
        self.multi_task = conf_train['multi_task']
        self.use_rnd = conf_train['use_rnd']
        conf_recurrence = config['recurrence']
        self.use_lstm = conf_recurrence['use_lstm']
        self.obs_space = obs_space

        # RND
        if self.use_rnd:
            # Only rnd the first dimension of the observation space
            self.rnd = RND(config, obs_space[0].shape)
            for p in self.rnd.target_net.parameters():
                p.requires_grad = False

        # Observation feature extraction
        if isinstance(obs_space, (gym_spaces.Tuple, gymnasium_spaces.Tuple)):
            # UGV
            if not self.multi_task and obs_space[0].shape == (84, 84, 3) and obs_space[1].shape == (400,):
                self.obs_net = ObsNetUGV(obs_space)
                in_features_size = self.obs_net.output_size
            # UGV with Task ID
            elif self.multi_task and obs_space[0].shape == (84, 84, 3) and obs_space[1].shape == (400,):
                self.obs_net = ObsNetUGV(obs_space)
                in_features_size = self.obs_net.output_size
                self.task_num = len(self.config.get('task', []))
                self.task_net = TaskNet(self.obs_space[-1].shape[0], 16)
                self.task_feature_size = self.task_net.output_size
                self.task_predict_net = TaskPredictNet(self.hidden_layer_size, 64, self.task_num)
            # Simple Vector With Task ID shape like((17,), (4,))
            elif self.multi_task and len(obs_space[0].shape) == 1 and len(obs_space[1].shape) == 1:
                self.task_num = len(self.config.get('task', []))
                self.task_net = TaskNet(self.obs_space[1].shape[0], 16)
                self.task_feature_size = self.task_net.output_size
                self.task_predict_net = TaskPredictNet(self.hidden_layer_size, 64, self.task_num)
                in_features_size = self.obs_space[0].shape[0]
            else:
                raise NotImplementedError(obs_space)
        # simple vector
        elif len(obs_space.shape) == 1:
            in_features_size = obs_space.shape[0]
        # single image
        elif len(obs_space.shape) == 3:
            self.obs_net = ObsNetImage(obs_space)
            in_features_size = self.obs_net.output_size
        else:
            raise NotImplementedError(obs_space.shape)

        # rnn
        if self.use_lstm:
            self.rnn_net = RNN(config, in_features_size)
            after_rnn_size = self.rnn_net.output_size
        else:
            after_rnn_size = in_features_size

        # hidden layer: out shape(self.hidden_layer_size,)
        self.hidden_net = HiddenNet(after_rnn_size, self.hidden_layer_size)

        # actor: in shape(self.hidden_layer_size,)
        if self.multi_task:
            self.actor = MultiGaussianActor(config, self.hidden_layer_size, action_space, self.task_num)
            # self.critic = MultiCritic(self.hidden_layer_size, 1, config=config, task_num=self.task_num)
            self.critic = Critic(config, self.hidden_layer_size + self.task_feature_size, 1)
        else:
            self.actor = GaussianActor(config, self.hidden_layer_size, action_space)
            self.critic = Critic(config, self.hidden_layer_size, 1)

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
                task_feature = self.task_net(obs[-1])
                feature = obs[0] if len(obs[0].shape) == 2 else self.obs_net(obs[:-1])
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
            task_predict = self.task_predict_net(feature)
            # critic
            value, rnd_value = self.critic(torch.cat((feature, task_feature), -1))
        else:
            # actor
            dist = self.actor(feature)
            task_predict = None
            # critic
            value, rnd_value = self.critic(feature)
        return dist, value, rnd_value, hidden_out, task_predict

    def eval_forward(self, obs, hidden_in: torch.Tensor = None, sequence_length: int = 1, module_index: int = -1):
        '''
        Only use for test no training: auto select actor to forward
        Args:
            state {tensor, list} -- observation tensor
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
        Returns:
            {dist}: action dist
            {value}: value base on current state
            {hidden_out}: RNN hidden out feature  
        '''
        # complex input or image or multi_task(obs and task id)
        if isinstance(self.obs_space, (gymnasium_spaces.Tuple, gym_spaces.Tuple)) or len(self.obs_space.shape) == 3:
            if self.multi_task:
                feature = obs[0] if len(obs[0].shape) == 2 else self.obs_net(obs[:-1])
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
            if module_index == -1:
                task_predict = self.task_predict_net(feature)
                module_index = torch.argmax(task_predict).item()
                print(module_index)
            # select actor
            dist = self.actor(feature, module_index)
        else:
            # actor
            dist = self.actor(feature)

        return dist, hidden_out
