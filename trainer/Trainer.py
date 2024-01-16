import signal
from datetime import datetime

import numpy as np
import torch
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces
from tqdm import tqdm

from algorithm.PPO import PPO
from normalization.RewardScaling import RewardScaling
from normalization.RNDRewardScaling import RNDRewardScaling
from normalization.StateNormalizer import StateNormalizer
from replaybuffer.Buffer import Buffer
from utils.ConfigHelper import ConfigHelper
from utils.env_helper import create_env
from utils.Logger import Logger
from utils.obs_2_tensor import _obs_2_tensor
from utils.polynomial_decay import get_decay
from utils.recurrent_cell_init import recurrent_cell_init
from worker.Worker import Worker
from worker.WorkerCommand import WorkerCommand


class Trainer:
    """
    1. Read the config file and checking
    2. Listen for keyboard interrupts to save the model
    3. Obtain the action space and observation space
    4. Init reward scaling
    5. Init buffer
    6. Init model and optimizer
    7. Init environment workers
    8. Random seed
    9. Reset workers
    10. Set log files
    11. Check resume
    12. Starting training
    """

    def __init__(self, config_file: str) -> None:
        """
        Args:
            config_file {str} -- path of yaml file that save the parameters
        """
        print("Step 1: Read the config file and checking")
        self.conf = ConfigHelper(config_file)

        print("Step 2: Listen for keyboard interrupts to save the model")
        signal.signal(signal.SIGINT, self._signal_handler)

        print("Step 3: Obtain the action space and observation space")
        _dummy_env = create_env(self.conf)
        self.obs_space = _dummy_env.observation_space
        self.action_space = (
            _dummy_env.action_space
            if self.conf.env_action_type == "continuous"
            else _dummy_env.action_space.n
        )
        _dummy_env.close()

        print("Step 4: Init reward scaling and state normalizer")
        self.reward_scaling = [
            RewardScaling(1, 0.99, self.conf) for _ in range(self.conf.num_workers)
        ]
        self.rnd_scaling = [
            RNDRewardScaling(shape=(1,), config=self.conf)
            for _ in range(self.conf.task_num)
        ]
        self.state_normalizer = StateNormalizer(self.obs_space, self.conf)

        print("Step 5: Init buffer")
        self.buffer = [
            Buffer(self.conf, self.obs_space, self.action_space)
            for _ in range(self.conf.task_num)
        ]

        print("Step 6: Init model and optimizer")
        self.ppo_agent = PPO(self.obs_space, self.action_space, self.conf)

        print("Step 7: Init environment workers")
        self.workers = [Worker(self.conf, w) for w in range(self.conf.num_workers)]

        print("Step 8: Random seed")
        if self.conf.random_seed != 0:
            torch.manual_seed(self.conf.random_seed)
            np.random.seed(self.conf.random_seed)

        print("Step 9: Reset workers")
        self.obs, self.recurrent_cell = self._reset_env()

        print("Step 10: Set log files")
        self.logger = Logger(self.conf)
        self.conf.run_num = self.logger.run_num

        print("Step 11: Check resume")
        if self.conf.resume:
            self.ppo_agent.load(self.logger.latest_checkpoint)
            self.reward_scaling = self.logger.load_pickle("reward_scaling.pkl")
            self.rnd_scaling = self.logger.load_pickle("rnd_scaling.pkl")
            self.state_normalizer = self.logger.load_pickle("state_normalizer.pkl")

    def run(self):
        """
        sample data --> prepare batch data --> gengerate mini batch data --> update PPO
        """
        print("Step 12: Starting training")
        self.start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)

        for self.conf.update in range(self.conf.update, self.conf.max_updates + 1):
            # Parameter decay
            learning_rate, clip_range, entropy_coeff, task_coeff = get_decay(self.conf)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            for b in self.buffer:
                b.prepare_batch_dict()

            # train K epochs
            (
                actor_losses,
                critic_losses,
                total_losses,
                dist_entropys,
                task_losses,
                rnd_losses,
            ) = [[] for _ in range(6)]
            for _ in range(self.conf.K_epochs):
                mini_batch_generator = self._multi_buff_mini_batch_generator()
                for mini_batch in mini_batch_generator:
                    losses = self.ppo_agent.train_mini_batch(
                        learning_rate,
                        clip_range,
                        entropy_coeff,
                        task_coeff,
                        mini_batch,
                        self.actual_sequence_length,
                    )
                    actor_losses.append(losses[0])
                    critic_losses.append(losses[1])
                    total_losses.append(losses[2])
                    dist_entropys.append(losses[3])
                    if losses[4] is not None:
                        task_losses.append(losses[4])
                    if losses[5] is not None:
                        rnd_losses.append(losses[5])

            # write logs
            self.conf.i_episode += len(sampled_episode_info)
            episode_result = Trainer._process_episode_info(sampled_episode_info)
            self.logger.write_tensorboard(
                (
                    actor_losses,
                    critic_losses,
                    total_losses,
                    dist_entropys,
                    task_losses,
                    rnd_losses,
                    learning_rate,
                    clip_range,
                    entropy_coeff,
                    task_coeff,
                    episode_result,
                    np.mean([b.rewards for b in self.buffer]),
                    np.mean([b.rnd_rewards for b in self.buffer])
                    if self.conf.use_rnd
                    else None,
                ),
                self.conf.update * self.conf.num_workers * self.conf.worker_steps,
            )
            self.logger.write_reward(
                self.conf.update,
                self.conf.update * self.conf.num_workers * self.conf.worker_steps,
                self.conf.i_episode,
                episode_result,
            )

            # save model weights
            if (
                self.conf.update != 0
                and self.conf.update % self.conf.save_model_freq == 0
            ):
                self._save()

            # free memory
            [b.free_memory() for b in self.buffer]

    def _multi_buff_mini_batch_generator(self):
        mini_batch_generator = [b.recurrent_mini_batch_generator() for b in self.buffer]
        self.actual_sequence_length = self.buffer[0].actual_sequence_length
        while True:
            try:
                mini_batch = [next(mg) for mg in mini_batch_generator]
                yield mini_batch
            except StopIteration:
                break

    def _reset_env(self):
        if self.conf.use_state_normailzation:
            print("---Pre_normalization started.---")
            total_pre_normalization_steps = 512
            for worker in self.workers:
                worker.child.send((WorkerCommand.reset, None))
            for worker in self.workers:
                o = worker.child.recv()
                self.state_normalizer(o)
            for _ in tqdm(range(total_pre_normalization_steps)):
                for worker in self.workers:
                    worker.child.send((WorkerCommand.step, self.action_space.sample()))
                for worker in self.workers:
                    o, _, _, info = worker.child.recv()
                    self.state_normalizer(o)
                    if info:
                        worker.child.send((WorkerCommand.reset, None))
                        o = worker.child.recv()
                        self.state_normalizer(o)
            print("---Pre_normalization is done.---")

        """ reset all environment in workers """
        assert isinstance(self.obs_space, (gym_spaces.Tuple, gymnasium_spaces.Tuple))
        obs = [
            np.zeros((self.conf.num_workers,) + _o.shape, dtype=np.float32)
            for _o in self.obs_space
        ]

        for index, worker in enumerate(self.workers):
            worker.child.send((WorkerCommand.reset, None))
            o = worker.child.recv()
            o = self.state_normalizer(o)
            for _obs_item, _o_item in zip(obs, o):
                _obs_item[index, :] = _o_item

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        cell = recurrent_cell_init(
            self.conf.num_workers,
            self.conf.hidden_state_size,
            self.conf.layer_type,
            self.conf.device,
        )

        # reset reward scaling
        for reward_sacling in self.reward_scaling:
            reward_sacling.reset()

        return obs, cell

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.
        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []

        for step in range(self.conf.worker_steps):
            self._sample_one_step(step, episode_infos)

        # Calculate internal reward
        if self.conf.use_rnd:
            next_states = torch.cat([b.rnd_next_obs for b in self.buffer], dim=0)
            total_rnd_rewards = self.ppo_agent.policy.rnd.calculate_rnd_rewards(
                next_states
            )
            for index in range(self.conf.task_num):
                buffer = self.buffer[index]
                start = index * self.conf.worker_per_task
                end = start + self.conf.worker_per_task
                slice_range = slice(start, end)

                # save rnd rewards to buffer
                buffer.rnd_rewards = self.rnd_scaling[index].normalize_rnd_rewards(
                    total_rnd_rewards[slice_range]
                )

        # Calculate advantages
        last_state_t = _obs_2_tensor(self.obs, self.conf.device)
        (
            _,
            _,
            last_value_t,
            last_rnd_value_t,
            _,
        ) = self.ppo_agent.select_action(last_state_t, self.recurrent_cell)
        for index in range(self.conf.task_num):
            buffer = self.buffer[index]
            start = index * self.conf.worker_per_task
            end = start + self.conf.worker_per_task
            slice_range = slice(start, end)

            buffer.calc_advantages(
                last_value_t[slice_range],
                last_rnd_value_t[slice_range] if last_rnd_value_t is not None else None,
            )

        return episode_infos

    def _sample_one_step(
        self,
        step: int,
        episode_infos: list,
    ):
        # numpy observation to tensor
        state_t = _obs_2_tensor(self.obs, self.conf.device)
        for index in range(self.conf.task_num):
            buffer = self.buffer[index]
            start = index * self.conf.worker_per_task
            end = start + self.conf.worker_per_task
            slice_range = slice(start, end)
            for obs_item, state_item in zip(buffer.obs, state_t):
                obs_item[:, step, :] = state_item[slice_range, :]

            if self.conf.use_lstm:
                if self.conf.layer_type == "gru":
                    buffer.hxs[:, step] = self.recurrent_cell.squeeze(0)[slice_range, :]
                elif self.conf.layer_type == "lstm":
                    buffer.hxs[:, step] = self.recurrent_cell[0].squeeze(0)[
                        slice_range, :
                    ]
                    buffer.cxs[:, step] = self.recurrent_cell[1].squeeze(0)[
                        slice_range, :
                    ]

        # Gradients can be omitted for sampling training data
        with torch.no_grad():
            # forwad model
            (
                action_t,
                action_logprob_t,
                value_t,
                rnd_value_t,
                self.recurrent_cell,
            ) = self.ppo_agent.select_action(state_t, self.recurrent_cell)

        # save to buffer
        for index in range(self.conf.task_num):
            buffer = self.buffer[index]
            start = index * self.conf.worker_per_task
            end = start + self.conf.worker_per_task
            slice_range = slice(start, end)

            buffer.actions[:, step] = action_t[slice_range]
            buffer.log_probs[:, step] = action_logprob_t[slice_range]
            buffer.values[:, step] = value_t[slice_range]
            if self.conf.use_rnd:
                buffer.rnd_values[:, step] = rnd_value_t[slice_range]

        # Send actions to the environments
        for worker, action in zip(self.workers, action_t):
            worker.child.send((WorkerCommand.step, action.cpu().numpy()))

        # Retrieve step results from the environments
        for worker_index, worker in enumerate(self.workers):
            obs_w, reward_w, done_w, info = worker.child.recv()
            obs_w = self.state_normalizer(obs_w)
            reward_w = self.reward_scaling[worker_index](reward_w)

            buffer = self.buffer[worker_index // self.conf.worker_per_task]
            buffer_index = worker_index % self.conf.worker_per_task
            buffer.rewards[buffer_index, step] = reward_w
            buffer.dones[buffer_index, step] = done_w

            if info:
                # Store the information of the completed episode (e.g. total reward, episode length)
                episode_infos.append(info)
                # Reset agent (potential interface for providing reset parameters)
                worker.child.send((WorkerCommand.reset, None))
                obs_w = worker.child.recv()
                # Reset recurrent cell states
                if self.conf.use_lstm and self.conf.reset_hidden_state:
                    rc = recurrent_cell_init(
                        1,
                        self.conf.hidden_state_size,
                        self.conf.layer_type,
                        self.conf.device,
                    )
                    if self.conf.layer_type == "lstm":
                        self.recurrent_cell[0][:, worker_index, :] = rc[0]
                        self.recurrent_cell[1][:, worker_index, :] = rc[1]
                    elif self.conf.layer_type == "gru":
                        self.recurrent_cell[:, worker_index, :] = rc
                # reset reward scaling
                self.reward_scaling[worker_index].reset()

            # Store latest observations
            for _obs, _o in zip(self.obs, obs_w):
                _obs[worker_index, :] = _o

        # save next obs in buffer for rnd TODO !!! if done is true, next obs is not used
        _state_t = _obs_2_tensor(self.obs, self.conf.device)
        if self.conf.use_rnd:
            for index in range(self.conf.task_num):
                buffer = self.buffer[index]
                start = index * self.conf.worker_per_task
                end = start + self.conf.worker_per_task
                slice_range = slice(start, end)
                buffer.rnd_next_obs[:, step, :] = _state_t[0][slice_range, :]

    @staticmethod
    def _process_episode_info(episode_info: list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.
        Args:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(
                        episode_result
                    )
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def close(self, done: bool = True):
        """close traning.
        Args:
            done {bool} -- Whether the maximum number of training steps has been reached
        """
        if not done:
            self._save()
        self.logger.close()

        for worker in self.workers:
            worker.child.send((WorkerCommand.close, None))

        print(
            "============================================================================================"
        )
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - self.start_time)
        print(
            "============================================================================================"
        )

    def _save(self):
        """save model & reward scaling & yaml file"""

        # save model
        print(
            "--------------------------------------------------------------------------------------------"
        )
        self.logger.save_checkpoint(self.conf.update, self.ppo_agent)

        # save reward scaling
        self.logger.save_pickle(self.reward_scaling, "reward_scaling.pkl")
        self.logger.save_pickle(self.rnd_scaling, "rnd_scaling.pkl")
        self.logger.save_pickle(self.state_normalizer, "state_normalizer.pkl")
        # save yaml file
        self.conf.save(self.logger.run_log_dir)
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def _signal_handler(self, sig, frame):
        """save when keyboard interrupt"""
        self.close(done=False)
