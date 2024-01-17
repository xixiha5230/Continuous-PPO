import numpy as np
import torch

from algorithm.ActorCritic import ActorCritic
from utils.ConfigHelper import ConfigHelper


class PPO:
    """PPO algorithm"""

    def __init__(self, obs_space: tuple, action_space: tuple, config: ConfigHelper):
        """
        Args:
            obs_space {tuple} -- observation space
            action_space {tuple} -- action space
            config {dict} -- config dictionary
        """
        self.use_lstm = config.use_lstm
        self.layer_type = config.layer_type
        self.hidden_state_size = config.hidden_state_size

        self.vf_loss_coeff = config.vf_loss_coeff

        self.action_type = config.env_action_type
        self.device = config.device
        self.multi_task = config.multi_task
        self.use_rnd = config.use_rnd
        self.rnd_rate = config.rnd_rate

        self.use_drq = config.use_drq

        self.policy = ActorCritic(obs_space, action_space, config).to(self.device)
        if self.multi_task:
            self.task_predict_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.lr_schedule["init"], eps=1e-5
        )

    def eval_select_action(
        self,
        obs: torch.Tensor,
        hidden_in: torch.Tensor = None,
        module_index: int = -1,
        is_ros: bool = False,
    ):
        """Only use for test no training: select action based on state and hidden_in

        Args:
            obs {tensor} -- observations
            hidden_in {torch.Tensor} -- RNN hidden in feature
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {tensor}: action tensor
            {tensor}: RNN hidden out feature tensor
        """
        with torch.no_grad():
            dist, hidden_out, task_predict = self.policy.eval_forward(
                obs,
                hidden_in,
                sequence_length=1,
                module_index=module_index,
                is_ros=is_ros,
            )
            action = dist.sample().detach()
            if hidden_out is not None:
                hidden_out = (
                    hidden_out.detach()
                    if self.layer_type == "gru"
                    else tuple(map(lambda x: x.detach(), hidden_out))
                )
            return action, hidden_out, task_predict

    def select_action(
        self, obs: torch.Tensor, hidden_in: torch.Tensor = None, module_index: int = -1
    ):
        """Select action based on state and hidden_in

        Args:
            obs {array} -- observations
            hidden_in {torch.Tensor} -- RNN hidden in feature
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {tensor}: action tensor
            {tensor}: action logprob tensor
            {tensor}: value of current state tensor
            {tensor}: rnd value of current state tensor
            {tensor}: RNN hidden out feature tensor
        """
        with torch.no_grad():
            dist, value, rnd_value, hidden_out, _ = self.policy.forward(
                obs, hidden_in, sequence_length=1, module_index=module_index
            )
            action = dist.sample().detach()
            action_logprob = dist.log_prob(action).detach()
            value = value.detach()
            rnd_value = rnd_value.detach() if rnd_value != None else None
            if hidden_out is not None:
                hidden_out = (
                    hidden_out.detach()
                    if self.layer_type == "gru"
                    else tuple(map(lambda x: x.detach(), hidden_out))
                )
            return action, action_logprob, value, rnd_value, hidden_out

    def evaluate(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_in: torch.Tensor,
        sequence_length: int,
        module_index: int = 0,
    ):
        """evaluate current batch base on state and action

        Args:
            obs {torch.Tensor, list} -- observations
            action {torch.Tensor} -- old action
            hidden_in {torch.Tensor} -- RNN hidden in feature
            sequence_length {int} -- RNN sequence length
            module_index {int} -- index of Actor or Critic to select
        Returns:
            {tensor}: action logprob tensor
            {tensor}: value base on state and new action  and old action
            {tensor}: rnd value
            {tensor}: action entropy
            {tensor}: task predict probability
        """
        dist, value, rnd_value, _, task_predict = self.policy.forward(
            obs, hidden_in, sequence_length, module_index=module_index
        )
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprob, value, rnd_value, dist_entropy, task_predict

    def train_mini_batch(
        self,
        learning_rate: float,
        clip_range: float,
        entropy_coeff: float,
        task_coeff: float,
        mini_batch: dict,
        sequence_length: int,
    ) -> list:
        """Uses one mini batch to optimize the model.
        Args:
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            entropy_coeff {float} -- Current entropy bonus coefficient
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            sequence_length {int} -- RNN sequence length
        Returns:
            {tuple} -- tuple of trainig statistics (e.g. loss)
        """

        multi_task_results = [
            self._train_mini_batch(
                clip_range,
                entropy_coeff,
                task_coeff,
                task_mini_batch,
                sequence_length,
                mudule_index,
            )
            for mudule_index, task_mini_batch in enumerate(mini_batch)
        ]
        multi_task_results = list(zip(*multi_task_results))
        # TODO maybe use sum(x) * (1.0 / len(x))
        policy_loss, vf_loss, entropy_bonus, task_loss, rnd_loss, loss = map(
            lambda x: sum(x), multi_task_results
        )

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

    def _train_mini_batch(
        self,
        clip_range: float,
        entropy_coeff: float,
        task_coeff: float,
        mini_batch: dict,
        sequence_length: int,
        module_index: int = 0,
    ):
        """Uses one mini batch to optimize the model.
        Args:
            clip_range {float} -- Current clip range
            entropy_coeff {float} -- Current entropy bonus coefficient
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            sequence_length {int} -- RNN sequence length
        Returns:
            {tuple} -- tuple of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.use_lstm:
            recurrent_cell = (
                mini_batch["hxs"].unsqueeze(0)
                if self.layer_type == "gru"
                else (mini_batch["hxs"].unsqueeze(0), mini_batch["cxs"].unsqueeze(0))
            )

        # Forward
        (
            new_logprobs,
            new_values,
            new_rnd_value,
            dist_entropy,
            task_pridcit,
        ) = self.evaluate(
            mini_batch["obs"],
            mini_batch["actions"],
            recurrent_cell,
            sequence_length,
            module_index,
        )

        # Remove paddings
        new_values = new_values[mini_batch["loss_mask"]]
        new_logprobs = new_logprobs[mini_batch["loss_mask"]]
        dist_entropy = dist_entropy[mini_batch["loss_mask"]]
        task_pridcit = (
            task_pridcit[mini_batch["loss_mask"]]
            if isinstance(task_pridcit, torch.Tensor)
            else None
        )
        new_rnd_value = (
            new_rnd_value[mini_batch["loss_mask"]]
            if isinstance(new_rnd_value, torch.Tensor)
            else None
        )

        # Get normalized advantage
        normalized_advantage = mini_batch["normalized_advantages"]
        normalized_advantage = (
            normalized_advantage.unsqueeze(-1)
            if self.action_type == "continuous"
            else normalized_advantage
        )
        if self.use_rnd:
            normalized_rnd_advantage = mini_batch["normalized_rnd_advantages"]
            normalized_rnd_advantage = (
                normalized_rnd_advantage.unsqueeze(-1)
                if self.action_type == "continuous"
                else normalized_rnd_advantage
            )
            normalized_advantage = (
                normalized_advantage + self.rnd_rate * normalized_rnd_advantage
            )

        # Policy loss
        ratio = torch.exp(new_logprobs - mini_batch["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = (
            torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            * normalized_advantage
        )
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value function loss
        sampled_return = mini_batch["values"] + mini_batch["advantages"]
        clipped_value = mini_batch["values"] + (
            new_values - mini_batch["values"]
        ).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max(
            (new_values - sampled_return) ** 2, (clipped_value - sampled_return) ** 2
        )
        if self.use_rnd:
            rnd_return = mini_batch["rnd_values"] + mini_batch["rnd_advantages"]
            rnd_clipped_value = mini_batch["rnd_values"] + (
                new_rnd_value - mini_batch["rnd_values"]
            ).clamp(min=-clip_range, max=clip_range)
            rnd_vf_loss = torch.max(
                (new_rnd_value - rnd_return) ** 2, (rnd_clipped_value - rnd_return) ** 2
            )
            vf_loss = 0.5 * (vf_loss + rnd_vf_loss)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = dist_entropy.mean()

        # RND loss
        rnd_loss = (
            self.policy.rnd.calculate_rnd_loss(mini_batch["rnd_next_obs"])
            if self.use_rnd
            else 0
        )

        # Task predictor loss
        if self.multi_task:
            # TODO 不argmax行不行
            # task_label =  mini_batch['obs'][-1][mini_batch['loss_mask']]
            task_label = torch.argmax(
                mini_batch["obs"][-1][mini_batch["loss_mask"]], dim=-1
            )
            task_loss = self.task_predict_loss(task_pridcit, task_label)
            # task_loss = 0
        else:
            task_loss = 0

        # Complete total loss
        total_loss = (
            -policy_loss
            + self.vf_loss_coeff * vf_loss
            - entropy_coeff * entropy_bonus
            + task_loss * task_coeff
        )
        return policy_loss, vf_loss, entropy_bonus, task_loss, rnd_loss, total_loss

    def save(self, checkpoint_path: str):
        """save old policy state dict
        Args:
            checkpoint_path {str} -- checkpoint path
        """
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        """load policy state dict
        Args:
            checkpoint_path {str} -- checkpoint path
        """
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
