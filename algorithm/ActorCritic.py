import torch
import torch.nn as nn
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces

from algorithm.RND import RND
from layers.Actor import GaussianActor, MultiGaussianActor
from layers.Critic import Critic
from layers.Hidden import HiddenNet
from layers.RNN import RNN
from layers.StateNet import ObsNetImage, ObsNetUGV, ObsNetUGVNew
from layers.TaskNet import TaskNet, TaskPredictNet
from utils.ConfigHelper import ConfigHelper


class ActorCritic(nn.Module):
    """Actor Critic Module"""

    def __init__(self, obs_space, action_space, config: ConfigHelper):
        """
        Args:
            obs_space {tuple} -- observation space
            action_space {tuple} -- action space
            config {dict} -- config dictionary
        """
        super(ActorCritic, self).__init__()
        self.hidden_layer_size = config.hidden_layer_size
        self.multi_task = config.multi_task
        self.use_rnd = config.use_rnd
        self.use_lstm = config.use_lstm
        self.obs_space = obs_space

        # Observation feature extraction
        if (
            len(obs_space) >= 2
            and obs_space[0].shape == (84, 84, 3)
            and obs_space[1].shape == (400,)
        ):
            # UGV
            self.obs_net = ObsNetUGV(obs_space)
        elif (
            len(obs_space) >= 2
            and obs_space[0].shape == (84, 84, 3)
            and obs_space[1].shape == (802,)
            and obs_space[2].shape == (2,)
        ):
            # image and vector
            self.obs_net = ObsNetUGVNew(obs_space)
        elif len(obs_space) == 1 and obs_space[0].shape == (84, 84, 3):
            # single image
            self.obs_net = ObsNetImage(obs_space[0])
        elif len(obs_space) == 1:
            # vector obs
            pass
        else:
            raise NotImplementedError(obs_space)
        in_features_size = (
            self.obs_net.output_size
            if hasattr(self, "obs_net")
            else obs_space[0].shape[0]
        )

        # RND
        if self.use_rnd:
            # TODO mybe can rnd rnn feature
            # Only rnd the first dimension of the observation space !!!
            self.rnd = RND(config, obs_space[0].shape)

        # Task net
        if self.multi_task:
            self.task_num = config.task_num
            assert self.task_num > 0
            self.task_net = TaskNet(self.obs_space[-1].shape[0], 16)
            self.task_feature_size = self.task_net.output_size
            self.task_predict_net = TaskPredictNet(
                self.obs_net.output_size, 128, self.task_num
            )

        # RNN
        if self.use_lstm:
            self.rnn_net = RNN(config, in_features_size)
        after_rnn_size = (
            self.rnn_net.output_size if hasattr(self, "rnn_net") else in_features_size
        )

        # Hidden layer
        self.hidden_net = HiddenNet(after_rnn_size, self.hidden_layer_size)

        # Actor Critic
        if self.multi_task:
            self.actor = MultiGaussianActor(
                config, self.hidden_layer_size, action_space, self.task_num
            )
            self.critic = Critic(
                config, self.hidden_layer_size + self.task_feature_size, 1
            )
        else:
            self.actor = GaussianActor(config, self.hidden_layer_size, action_space)
            self.critic = Critic(config, self.hidden_layer_size, 1)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_in: torch.Tensor = None,
        sequence_length: int = 1,
        module_index: int = -1,
        is_ros: bool = False,
    ):
        """
        Args:
            obs {tensor, list} -- observation tensor
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {distribution}: action distribution
            {tensor}: value base on current state
            {tensor}: rnd value base on current state
            {tensor}: RNN hidden out feature
            {task_predict}: predicted task id
        """

        feature = self._forward_obs(obs, is_ros)
        obs_feature = feature

        feature, hidden_out = self._forward_rnn(feature, hidden_in, sequence_length)
        feature = self._forward_hidden(feature)
        task_predict = None
        if self.multi_task:
            task_feature = self.task_net(obs[-1])
            task_predict = self.task_predict_net(obs_feature)
            dist = self.forward_actor((task_predict, feature))
            value, rnd_value = self.forward_critic(
                torch.cat((feature, task_feature), -1)
            )
        else:
            dist = self.forward_actor(feature)
            value, rnd_value = self.forward_critic(feature)
        return dist, value, rnd_value, hidden_out, task_predict

    def eval_forward(
        self,
        obs: torch.Tensor,
        hidden_in: torch.Tensor = None,
        sequence_length: int = 1,
        module_index: int = -1,
        is_ros: bool = False,
    ):
        """
        Only use for test no training: auto select actor to forward
        Args:
            obs {tensor, list} -- observation tensor
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {dist}: action dist
            {hidden_out}: RNN hidden out feature
        """
        feature = self._forward_obs(obs, is_ros)
        obs_feature = feature
        feature, hidden_out = self._forward_rnn(feature, hidden_in, sequence_length)
        feature = self._forward_hidden(feature)

        task_predict = None
        if self.multi_task:
            task_predict = self.task_predict_net(obs_feature)
            # module_index = torch.argmax(task_predict).item()
            # print(task_predict)
            dist = self.forward_actor((task_predict, feature))
        else:
            dist = self.forward_actor(feature)
        return dist, hidden_out, task_predict

    def forward_obs(self, obs: torch.Tensor, is_ros: bool = False):
        """
        Args:
            obs {tensor, list} -- observation tensor
        Returns:
            {tensor}: feature
        """
        feature = self._forward_obs(obs, is_ros)
        feature, hidden_out = self._forward_rnn(feature)
        feature = self._forward_hidden(feature)
        return feature, hidden_out

    def forward_critic(self, feature: torch.Tensor):
        """
        Args:
            feature {tensor} -- feature tensor
        Returns:
            {tensor}: value base on current state
            {tensor}: rnd value base on current state
        """
        value, rnd_value = self.critic(feature)
        return value, rnd_value

    def forward_actor(self, feature: torch.Tensor):
        """
        Args:
            feature {tensor} -- feature tensor
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {distribution}: action distribution
        """
        dist = self.actor(feature)
        return dist

    def _forward_obs(self, obs: torch.Tensor, is_ros: bool = False):
        """
        Args:
            obs {tensor, list} -- observation tensor
        Returns:
            {tensor}: feature
        """
        if self.multi_task:
            feature = obs[:-1]
        else:
            feature = obs
        feature = self.obs_net(feature, is_ros) if hasattr(self, "obs_net") else obs[0]
        return feature

    def _forward_rnn(
        self,
        feature: torch.Tensor,
        hidden_in: torch.Tensor = None,
        sequence_length: int = 1,
    ):
        """
        Args:
            feature {tensor} -- feature tensor
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
        Returns:
            {tensor}: feature
        """
        if self.use_lstm:
            feature, hidden_out = self.rnn_net(feature, hidden_in, sequence_length)
        else:
            hidden_out = None
        return feature, hidden_out

    def _forward_hidden(self, feature: torch.Tensor):
        """
        Args:
            feature {tensor} -- feature tensor
        Returns:
            {tensor}: feature
        """
        feature = self.hidden_net(feature)
        return feature
