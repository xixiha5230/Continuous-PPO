import numpy as np
import torch
from algorithm.ActorCritic import ActorCritic


class PPO:
    '''PPO algorithm'''

    def __init__(self, obs_space: tuple, action_space: tuple, config: dict):
        '''
        Args:
            obs_space {tuple} -- observation space
            action_space {tuple} -- action space
            config {dict} -- config dictionary
        '''
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
        self.action_type = self.conf_train['action_type']
        self.device = self.conf_train['device']
        self.multi_task = self.conf_train['multi_task']
        self.use_rnd = self.conf_train['use_rnd']

        self.policy = ActorCritic(obs_space, action_space, self.config).to(self.device)
        self.policy.train()
        self.policy.rnd.eval()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.conf_ppo['lr_schedule']['init'], eps=1e-5)

    def eval_select_action(self, obs, hidden_in: torch.Tensor = None, module_index: int = -1):
        ''' Only use for test no training: select action based on state and hidden_in

        Args:
            obs {array} -- observations
            hidden_in {torch.Tensor} -- RNN hidden in feature
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {action}: action tensor
            {action_logprob}: action logprob tensor
            {value}: value of current state tensor
            {hidden_out}: RNN hidden out feature tensor
        '''
        with torch.no_grad():
            dist, hidden_out = self.policy.eval_forward(
                obs, hidden_in, sequence_length=1, module_index=module_index)
            action = dist.sample().detach()
            if hidden_out is not None:
                if self.layer_type == 'lstm':
                    hidden_out = (hidden_out[0].detach(), hidden_out[1].detach())
                elif self.layer_type == 'gru':
                    hidden_out = hidden_out.detach()
                else:
                    raise NotImplementedError(self.layer_type)
            return action, hidden_out

    def select_action(self, obs, hidden_in: torch.Tensor = None, module_index: int = -1):
        ''' Select action based on state and hidden_in

        Args:
            obs {array} -- observations
            hidden_in {torch.Tensor} -- RNN hidden in feature
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {action}: action tensor
            {action_logprob}: action logprob tensor
            {value}: value of current state tensor
            {hidden_out}: RNN hidden out feature tensor
        '''
        with torch.no_grad():
            dist, value, rnd_value, hidden_out, _ = self.policy.forward(
                obs, hidden_in, sequence_length=1, module_index=module_index)
            action = dist.sample().detach()
            action_logprob = dist.log_prob(action).detach()
            value = value.detach()
            rnd_value = rnd_value.detach() if rnd_value != None else None
            if hidden_out is not None:
                if self.layer_type == 'lstm':
                    hidden_out = (hidden_out[0].detach(), hidden_out[1].detach())
                elif self.layer_type == 'gru':
                    hidden_out = hidden_out.detach()
                else:
                    raise NotImplementedError(self.layer_type)
            return action, action_logprob, value, rnd_value, hidden_out

    def evaluate(self, obs, action: torch.Tensor, hidden_in: torch.Tensor, sequence_length: int, module_index: int = 0):
        ''' evaluate current batch base on state and action

        Args:
            obs {torch.Tensor, list} -- observations
            action {torch.Tensor} -- old action
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {action_logprob}: action logprob tensor
            {value}: value base on state and new action  and old action
            {dist_entropy}: action entropy
        '''
        dist, value, rnd_value, _, task_predict = self.policy.forward(
            obs, hidden_in, sequence_length, module_index=module_index)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprob, value, rnd_value, dist_entropy, task_predict

    def train_mini_batch(self, learning_rate: float, clip_range: float, entropy_coeff: float, mini_batch: dict, sequence_length: int) -> list:
        '''Uses one mini batch to optimize the model.
        Args:
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            entropy_coeff {float} -- Current entropy bonus coefficient
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            sequence_length {int} -- RNN sequence length
        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        '''
        if self.multi_task:
            multi_task_results = [self._train_mini_batch(
                clip_range, entropy_coeff, task_mini_batch, sequence_length, mudule_index) for mudule_index, task_mini_batch in enumerate(mini_batch)]
            multi_task_results = list(zip(*multi_task_results))
            # not sum(x)/len(x) !!!
            policy_loss, vf_loss, entropy_bonus, task_loss, rnd_loss, loss = map(lambda x: sum(x), multi_task_results)
        else:
            policy_loss, vf_loss, entropy_bonus, task_loss, rnd_loss, loss = self._train_mini_batch(
                clip_range, entropy_coeff, mini_batch, sequence_length, -1)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()

        return (
            policy_loss.item(),
            vf_loss.item(),
            loss.item(),
            entropy_bonus.item(),
            task_loss.item() if task_loss != 0 else None,
            rnd_loss.item() if rnd_loss != 0 else None,
        )

    def _train_mini_batch(self, clip_range: float, entropy_coeff: float, mini_batch: dict, sequence_length: int, module_index: int = 0):
        '''Uses one mini batch to optimize the model.
        Args:
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            entropy_coeff {float} -- Current entropy bonus coefficient
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            sequence_length {int} -- RNN sequence length
        Returns:
            {torch.tensor} -- policy_loss
            {torch.tensor} -- vf_loss
            {torch.tensor} -- entropy_bonus
            {torch.tensor} -- total_loss
        '''
        # Retrieve sampled recurrent cell states to feed the model
        if self.use_lstm:
            if self.layer_type == 'gru':
                recurrent_cell = mini_batch['hxs'].unsqueeze(0)
            elif self.layer_type == 'lstm':
                recurrent_cell = (mini_batch['hxs'].unsqueeze(0), mini_batch['cxs'].unsqueeze(0))
        else:
            recurrent_cell = None
        new_logprobs, state_values, rnd_value, dist_entropy, task_pridcit = self.evaluate(
            mini_batch['obs'], mini_batch['actions'], recurrent_cell, sequence_length, module_index)

        # Remove paddings
        state_values = state_values[mini_batch['loss_mask']]
        new_logprobs = new_logprobs[mini_batch['loss_mask']]
        dist_entropy = dist_entropy[mini_batch['loss_mask']]
        if self.multi_task:
            task_pridcit = task_pridcit[mini_batch['loss_mask']]
        if self.use_rnd:
            rnd_value = rnd_value[mini_batch['loss_mask']]

        normalized_advantage = mini_batch['normalized_advantages']
        if self.use_rnd:
            normalized_rnd_advantage = mini_batch['normalized_rnd_advantages']
            # TODO 比例参数化
            rnd_rate = 0.5
            normalized_advantage = (1 - rnd_rate) * normalized_advantage + rnd_rate * normalized_rnd_advantage

        # TODO why
        if self.action_type == 'continuous':
            normalized_advantage = normalized_advantage.unsqueeze(-1)

        # policy_loss
        ratio = torch.exp(new_logprobs - mini_batch['log_probs'])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # vf_loss
        sampled_return = mini_batch['values'] + mini_batch['advantages']
        clipped_value = mini_batch['values'] + (state_values - mini_batch['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((state_values - sampled_return) ** 2, (clipped_value - sampled_return)**2)
        if self.use_rnd:
            rnd_return = mini_batch['rnd_values'] + mini_batch['rnd_advantages']
            rnd_clipped_value = mini_batch['rnd_values'] + \
                (rnd_value - mini_batch['rnd_values']).clamp(min=-clip_range, max=clip_range)
            rnd_vf_loss = torch.max((rnd_value - rnd_return) ** 2, (rnd_clipped_value - rnd_return)**2)
            vf_loss = 0.5 * (vf_loss + rnd_vf_loss)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = dist_entropy.mean()

        # RND loss
        if self.use_rnd:
            rnd_loss = self.policy.rnd.calculate_rnd_loss(mini_batch['rnd_next_obs'])
        else:
            rnd_loss = 0

        # task predict loss
        if self.multi_task:
            # task_pridcit = torch.argmax(task_pridcit, dim=0
            #                             ,keepdim=True)
            y = torch.argmax(mini_batch['obs'][-1][mini_batch['loss_mask']], dim=-1)
            task_loss = self.criterion(task_pridcit, y)
        else:
            task_loss = 0

        # Complete loss
        total_loss = -(policy_loss - self.vf_loss_coeff * vf_loss + entropy_coeff * entropy_bonus - task_loss - rnd_loss)
        return policy_loss, vf_loss, entropy_bonus, task_loss, rnd_loss, total_loss

    def save(self, checkpoint_path: str):
        ''' save old policy state dict
        Args:
            checkpoint_path {str} -- checkpoint path
        '''
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        ''' load policy state dict
        Args:
            checkpoint_path {str} -- checkpoint path
        '''
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        # self.old_policy.load_state_dict(self.policy.state_dict())

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

    @staticmethod
    def _state_2_tensor(state: list, device: str):
        '''state array to tensor
        Args:
            state {int} -- aka observation
            device {str} -- 'cuda' or 'cpu'
        Returns:
            {torch.Tensor} -- tensor of state
        '''
        if isinstance(state, list):
            # single state
            if not isinstance(state[0], list):
                state = [state]
            state = [torch.FloatTensor(np.array([s[i] for s in state])).to(device)
                     for i in range(len(state[0]))]
        else:
            if len(state.shape) == 1:
                state = [state]
            state = torch.FloatTensor(np.array(state)).to(device)
        return state
