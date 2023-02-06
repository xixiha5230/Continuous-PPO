import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithm.ActorCritic import ActorCritic
from replaybuffer.RolloutBuffer import RolloutBuffer


################################## PPO Policy ##################################
class PPO:
    def __init__(self, obs_space, action_dim, config):
        self.config = config
        self.conf_worker = config['worker']
        self.conf_recurrence = config['recurrence']
        self.conf_ppo = config['ppo']
        self.use_lstm = self.conf_recurrence['use_lstm']
        self.has_continuous_action_space = config['has_continuous_action_space']
        if self.has_continuous_action_space:
            self.action_std = config['action_std']

        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.K_epochs = config['K_epochs']
        self.buffer = RolloutBuffer()
        lr_actor = config['lr_actor']
        lr_critic = config['lr_critic']
        self.vf_loss_coeff = self.conf_ppo['vf_loss_coeff']
        self.entropy_coeff = self.conf_ppo['entropy_coeff']
        self.device = config['device']
        self.policy = ActorCritic(
            obs_space, action_dim, self.config).to(self.device)
        self.networks = [
            {'params': self.policy.state.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ]
        if self.use_lstm:
            self.networks.append(
                {'params': self.policy.rnn.parameters(), 'lr': lr_actor})
        self.optimizer = torch.optim.Adam(self.networks)
        self.policy_old = ActorCritic(
            obs_space, action_dim, self.config).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print(
                "--------------------------------------------------------------------------------------------")
            print(
                "WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print(
                "--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print(
                    "setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print(
                "WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, hidden_in=None):
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).to(self.device)
            action, action_logprob, hidden_out = self.policy_old.act(
                state, hidden_in)
            action_s = action.detach().cpu().numpy() \
                if self.has_continuous_action_space else action.item()
        return action_s, state, action, action_logprob, hidden_out

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
        if self.conf_recurrence['use_lstm']:
            if self.conf_recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.conf_recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(
                    0), samples["cxs"].unsqueeze(0))
        else:
            recurrent_cell = None
        action_logprobs, state_values, dist_entropy = self.policy.evaluate(
            samples["obs"], samples["actions"], recurrent_cell, self.conf_recurrence['sequence_length'])
        state_values = torch.squeeze(state_values)

        rewards = samples["advantages"]
        advantages = rewards - state_values.detach()

        ratio = torch.exp(action_logprobs - samples["log_probs"].detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 +
                            self.eps_clip) * advantages
        policy_loss = torch.min(surr1, surr2)
        policy_loss = self._masked_mean(
            policy_loss, samples["loss_mask"])

        # vf_loss = self.MseLoss(state_values, rewards)
        # vf_loss = (state_values - rewards) ** 2
        vf_loss = F.smooth_l1_loss(state_values, rewards)
        vf_loss = self._masked_mean(vf_loss, samples["loss_mask"])

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

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Returns the mean of the tensor but ignores the values specified by the mask.
        This is used for masking out the padding of the loss functions.
        Args:
            tensor {Tensor} -- The to be masked tensor
            mask {Tensor} -- The mask that is used to mask out padded values of a loss function
        Returns:
            {Tensor} -- Returns the mean of the masked tensor.
        """
        return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage))

    def init_recurrent_cell_states(self, num_sequences) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.
        Args:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hidden_state_size = self.conf_recurrence['hidden_state_size']
        layer_type = self.conf_recurrence["layer_type"]
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
