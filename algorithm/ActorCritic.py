import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_space, action_dim, config):
        super(ActorCritic, self).__init__()
        self.config = config

        self.has_continuous_action_space = config['has_continuous_action_space']
        self.device = config['device']
        self.use_lstm = config['recurrence']['use_lstm']
        if self.use_lstm:
            self.hidden_state_size = config['recurrence']['hidden_state_size']
        action_std_init = config['action_std']

        if self.has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full(
                (action_dim,), action_std_init * action_std_init).to(self.device)
        if len(obs_space.shape) == 1:
            # state
            self.state = nn.Sequential(
                nn.Linear(obs_space.shape[0], 64),
                nn.Tanh(),
                # nn.Linear(64, 64),
                # nn.Tanh(),
            )
        else:
            raise NotImplementedError(obs_space.shape)

        # gru
        if self.use_lstm:
            self.rnn = nn.GRU(64, self.hidden_state_size, batch_first=True)
            self.feature_dim = self.hidden_state_size
        else:
            self.feature_dim = 64
        # actor
        if self.has_continuous_action_space:
            self.actor = nn.Sequential(
                # nn.Linear(self.feature_dim, self.feature_dim),
                # nn.Tanh(),
                nn.Linear(self.feature_dim, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                # nn.Linear(self.feature_dim, self.feature_dim),
                # nn.Tanh(),
                nn.Linear(self.feature_dim, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            # nn.Linear(self.feature_dim, self.feature_dim),
            # nn.Tanh(),
            nn.Linear(self.feature_dim, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print(
                "--------------------------------------------------------------------------------------------")
            print(
                "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print(
                "--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, hidden_in=None, sequence_length=1):
        feature = self.state(state)
        if self.use_lstm:
            if sequence_length == 1:
                # Case: sampling training data or model optimization using sequence length == 1
                feature, hidden_out = self.rnn(
                    feature.unsqueeze(1), hidden_in)
                # Remove sequence length dimension
                feature = feature.squeeze(1)
            else:
                feature_shape = tuple(feature.size())
                feature = feature.reshape(
                    (feature_shape[0] // sequence_length), sequence_length, feature_shape[1])
                feature, hidden_out = self.rnn(
                    feature, hidden_in)
                out_shape = tuple(feature.size())
                feature = feature.reshape(
                    out_shape[0] * out_shape[1], out_shape[2])
        else:
            feature = feature
            hidden_out = None

        if self.has_continuous_action_space:
            action_mean = self.actor(feature)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(feature)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), hidden_out.detach() if hidden_out != None else None

    def evaluate(self, state, action, hidden_in=None, sequence_length=None):
        feature = self.state(state)
        if self.use_lstm:
            if sequence_length == 1:
                # Case: sampling training data or model optimization using sequence length == 1
                feature, _ = self.rnn(
                    feature.unsqueeze(1), hidden_in)
                # Remove sequence length dimension
                feature = feature.squeeze(1)
            else:
                feature_shape = tuple(feature.size())
                feature = feature.reshape(
                    (feature_shape[0] // sequence_length), sequence_length, feature_shape[1])
                feature, _ = self.rnn(
                    feature, hidden_in)
                out_shape = tuple(feature.size())
                feature = feature.reshape(
                    out_shape[0] * out_shape[1], out_shape[2])

        if self.has_continuous_action_space:
            action_mean = self.actor(feature)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(feature)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(feature)

        return action_logprobs, state_values, dist_entropy
