import numpy as np
import torch
from algorithm.ActorCritic import ActorCritic

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
        self.vf_loss_coeff = self.conf_ppo['vf_loss_coeff']
        self.conf_train = config['train']
        self.has_continuous_action = self.conf_train['has_continuous_action_space']
        self.device = self.conf_train['device']

        self.policy = ActorCritic(obs_space, action_space, self.config).to(self.device)
        self.old_policy = ActorCritic(obs_space, action_space, self.config).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.conf_ppo['lr_schedule']['init'], eps=1e-5)

    def select_action(self, state, hidden_in=None):
        with torch.no_grad():
            if isinstance(state, list):
                # single state
                if not isinstance(state[0], list):
                    state = [state]
                state = [torch.FloatTensor(np.array([s[i] for s in state])).to(self.device)
                         for i in range(len(state[0]))]
            else:
                if len(state.shape) == 1:
                    state = [state]
                state = torch.FloatTensor(np.array(state)).to(self.device)
            dist, value, hidden_out = self.old_policy.forward(state, hidden_in)
            action = dist.sample().detach()

            todo_action = action.cpu().numpy()
            value = value.squeeze(-1).detach()
            action_logprob = dist.log_prob(action).detach()
            hidden_out = hidden_out.detach() if hidden_out is not None else None

            return todo_action, state, action, action_logprob, value, hidden_out

    def evaluate(self, state, action, hidden_in, sequence_length):
        dist, value, hidden_out = self.policy.forward(state, hidden_in, sequence_length)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprob, value, dist_entropy

    def _train_mini_batch(self, learning_rate, clip_range, entropy_coeff, mini_batch: dict) -> list:
        '''Uses one mini batch to optimize the model.
        Args:
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            entropy_coeff {float} -- Current entropy bonus coefficient
            mini_batch {dict} -- The to be used mini batch data to optimize the model
        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        '''
        # Retrieve sampled recurrent cell states to feed the model
        if self.use_lstm:
            if self.layer_type == 'gru':
                recurrent_cell = mini_batch['hxs'].unsqueeze(0)
            elif self.layer_type == 'lstm':
                recurrent_cell = (mini_batch['hxs'].unsqueeze(0), mini_batch['cxs'].unsqueeze(0))
        else:
            recurrent_cell = None
        action_logprobs, state_values, dist_entropy = self.evaluate(
            mini_batch['obs'], mini_batch['actions'], recurrent_cell, self.sequence_length)
        # TODO 提前挤压
        state_values = torch.squeeze(state_values)

        # Remove paddings
        state_values = state_values[mini_batch['loss_mask']]
        action_logprobs = action_logprobs[mini_batch['loss_mask']]
        dist_entropy = dist_entropy[mini_batch['loss_mask']]

        normalized_advantage = (mini_batch['advantages'] - mini_batch['advantages'].mean()
                                ) / (mini_batch['advantages'].std() + 1e-8)
        # TODO 提前挤压
        if self.has_continuous_action:
            normalized_advantage = normalized_advantage.unsqueeze(-1)

        # policy_loss
        ratio = torch.exp(action_logprobs - mini_batch['log_probs'])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # vf_loss
        sampled_return = mini_batch['values'] + mini_batch['advantages']
        clipped_value = mini_batch['values'] + (state_values - mini_batch['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((state_values-sampled_return) ** 2, (clipped_value-sampled_return)**2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = dist_entropy.mean()

        # Complete loss
        loss = -(policy_loss - self.vf_loss_coeff * vf_loss + entropy_coeff * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        # TODO use policy_loss.cpu().data.numpy()
        return (policy_loss.detach().mean().item(),
                vf_loss.detach().mean().item(),
                loss.detach().mean().item(),
                entropy_bonus.detach().mean().item())

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.old_policy.load_state_dict(self.policy.state_dict())

    def init_recurrent_cell_states(self, num_sequences) -> tuple:
        '''Initializes the recurrent cell states (hxs, cxs) as zeros.
        Args:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        '''
        hidden_state_size = self.hidden_state_size
        layer_type = self.layer_type
        hxs = torch.zeros((num_sequences), hidden_state_size, dtype=torch.float32, device=self.device).unsqueeze(0)
        cxs = torch.zeros((num_sequences), hidden_state_size, dtype=torch.float32, device=self.device).unsqueeze(0)
        if layer_type == 'lstm':
            return hxs, cxs
        elif layer_type == 'gru':
            return hxs
        else:
            raise NotImplementedError(layer_type)
