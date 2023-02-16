import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.ActorCritic import ActorCritic
from gym import spaces

################################## PPO Policy ##################################


class PPO:
    def __init__(self, obs_space, action_space, config):
        self.config = config
        self.conf_worker = config['worker']

        self.conf_recurrence = config['recurrence']
        self.use_lstm = self.conf_recurrence['use_lstm']
        self.layer_type = self.conf_recurrence['layer_type']
        self.sequence_length = self.conf_recurrence['sequence_length']
        self.hidden_state_size = self.conf_recurrence['hidden_state_size']

        self.conf_ppo = config['ppo']
        self.gamma = self.conf_ppo['gamma']
        self.eps_clip = self.conf_ppo['eps_clip']
        lr = self.conf_ppo['lr']
        lr_std = self.conf_ppo['lr_std']
        self.vf_loss_coeff = self.conf_ppo['vf_loss_coeff']
        self.entropy_coeff = self.conf_ppo['entropy_coeff']

        self.conf_train = config['train']
        self.has_continuous_action = self.conf_train['has_continuous_action_space']
        self.device = self.conf_train['device']
        self.K_epochs = self.conf_train['K_epochs']

        self.policy = ActorCritic(
            obs_space, action_space, self.config).to(self.device)
        self.old_policy = ActorCritic(
            obs_space, action_space, self.config).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.networks = [
            {'params': self.policy.lin_hidden.parameters(), 'lr': lr},
            {'params': self.policy.mu.parameters(), 'lr': lr},
            {'params': self.policy.critic.parameters(), 'lr': lr}
        ]
        if isinstance(obs_space, spaces.Tuple) or len(obs_space.shape) == 3:
            self.networks.append(
                {'params': self.policy.state.parameters(), 'lr': lr})
        if self.has_continuous_action:
            self.networks.append(
                {'params': self.policy.sigma.parameters(), 'lr': lr_std})
        if self.use_lstm:
            self.networks.append(
                {'params': self.policy.rnn.parameters(), 'lr': lr})
        self.optimizer = torch.optim.Adam(self.networks, eps=1e-5)

    def select_action(self, state, hidden_in=None):
        with torch.no_grad():
            if isinstance(state, list):
                state = [torch.FloatTensor(np.array([s[i] for s in state])).to(self.device)
                         for i in range(len(state[0]))]
            else:
                state = torch.FloatTensor(np.array(state)).to(self.device)
            dist, value, hidden_out = self.old_policy.forward(
                state, hidden_in)
            value = value.squeeze(-1)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            action_s = action.detach().cpu().numpy()
            return action_s, state, action.detach(), action_logprob.detach(), \
                value.detach(), hidden_out.detach() if hidden_out is not None else None

    def evaluate(self, state, action, hidden_in, sequence_length):
        dist, value, hidden_out = self.policy.forward(
            state, hidden_in, sequence_length)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprob, value, dist_entropy

    def _train_mini_batch(self, samples: dict) -> list:
        """Uses one mini batch to optimize the model.
        Args:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient
        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.use_lstm:
            if self.layer_type == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.layer_type == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(
                    0), samples["cxs"].unsqueeze(0))
        else:
            recurrent_cell = None
        action_logprobs, state_values, dist_entropy = self.evaluate(
            samples["obs"], samples["actions"], recurrent_cell, self.sequence_length)
        state_values = torch.squeeze(state_values)

        advantages_unpaddedd = torch.masked_select(
            samples["advantages"], samples["loss_mask"])
        normalized_advantage = (
            samples["advantages"] - advantages_unpaddedd.mean()) / (advantages_unpaddedd.std() + 1e-8)
        if self.has_continuous_action:
            normalized_advantage = normalized_advantage.unsqueeze(-1)
        # rewards = samples["advantages"]
        # advantages = rewards - state_values.detach()
        # advantages = advantages.unsqueeze(-1)

        ratio = torch.exp(action_logprobs - samples["log_probs"].detach())
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 +
                            self.eps_clip) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = PPO._masked_mean(
            policy_loss, samples["loss_mask"])

        # vf_loss = (state_values - rewards) ** 2
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (state_values - samples["values"]).clamp(
            min=-self.eps_clip, max=self.eps_clip)
        vf_loss = torch.max((state_values-sampled_return) **
                            2, (clipped_value-sampled_return)**2)
        # vf_loss = F.smooth_l1_loss(state_values, rewards)
        vf_loss = PPO._masked_mean(vf_loss, samples["loss_mask"])

        # Entropy Bonus
        entropy_bonus = self._masked_mean(
            dist_entropy, samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss -
                 self.vf_loss_coeff * vf_loss +
                 self.entropy_coeff * entropy_bonus)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return (policy_loss.detach().mean().item(),
                vf_loss.detach().mean().item(),
                loss.detach().mean().item(),
                entropy_bonus.detach().mean().item())

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean of the tensor but ignores the values specified by the mask.
        This is used for masking out the padding of the loss functions.
        Args:
            tensor {Tensor} -- The to be masked tensor
            mask {Tensor} -- The mask that is used to mask out padded values of a loss function
        Returns:
            {Tensor} -- Returns the mean of the masked tensor.
        """
        tmp = tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1))
        return (tmp * mask).sum() / torch.clamp((torch.ones_like(tmp) * mask).float().sum(), min=1.0)

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))
        self.old_policy.load_state_dict(self.policy.state_dict())

    def init_recurrent_cell_states(self, num_sequences) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.
        Args:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hidden_state_size = self.hidden_state_size
        layer_type = self.layer_type
        hxs = torch.zeros(
            (num_sequences), hidden_state_size, dtype=torch.float32, device=self.device).unsqueeze(0)
        cxs = torch.zeros(
            (num_sequences), hidden_state_size, dtype=torch.float32, device=self.device).unsqueeze(0)
        if layer_type == "lstm":
            return hxs, cxs
        elif layer_type == 'gru':
            return hxs
        else:
            raise NotImplementedError(layer_type)
