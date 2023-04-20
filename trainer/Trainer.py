import signal
from datetime import datetime

import numpy as np
import torch
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces

from algorithm.PPO import PPO
from normalization.RewardScaling import RewardScaling
from replaybuffer.Buffer import Buffer
from utils.ConfigHelper import ConfigHelper
from utils.env_helper import create_env
from utils.Logger import Logger
from utils.obs_2_tensor import _obs_2_tensor
from utils.polynomial_decay import get_decay
from worker.Worker import Worker


class Trainer:
    '''
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
    '''

    def __init__(self, config_file: str) -> None:
        '''
        Args:
            config_file {str} -- path of yaml file that save the parameters
        '''
        print('Step 1: Read the config file and checking')
        self.conf = ConfigHelper(config_file)

        print('Step 2: Listen for keyboard interrupts to save the model')
        signal.signal(signal.SIGINT, self._signal_handler)

        print('Step 3: Obtain the action space and observation space')
        _dummy_env = create_env(self.conf)
        self.obs_space = _dummy_env.observation_space
        self.action_space = _dummy_env.action_space if self.conf.action_type == 'continuous' else _dummy_env.action_space.n
        _dummy_env.close()

        print('Step 4: Init reward scaling')
        if self.conf.use_reward_scaling:
            self.reward_scaling = [RewardScaling(1, 0.99) for _ in range(self.conf.num_workers)]

        print('Step 5: Init buffer')
        self.buffer = [Buffer(self.conf, self.obs_space, self.action_space) for _ in range(self.conf.task_num)]

        print('Step 6: Init model and optimizer')
        self.ppo_agent = PPO(self.obs_space, self.action_space, self.conf)

        print('Step 7: Init environment workers')
        self.workers = [Worker(self.conf, w) for w in range(self.conf.num_workers)]

        print('Step 8: Random seed')
        if self.conf.random_seed != 0:
            torch.manual_seed(self.conf.random_seed)
            np.random.seed(self.conf.random_seed)

        print('Step 9: Reset workers')
        self.obs, self.recurrent_cell = self._reset_env()

        print('Step 10: Set log files')
        self.logger = Logger(self.conf.env_name, self.conf.exp_name, self.conf.run_num, self.conf.resume)
        self.conf.run_num = self.logger.run_num

        print('Step 11: Check resume')
        if self.conf.resume:
            self.ppo_agent.load(self.logger.latest_checkpoint)
            if self.conf.use_reward_scaling:
                self.reward_scaling = self.logger.load_pickle('reward_scaling.pkl')

    def run(self):
        '''
        sample data --> prepare batch data --> gengerate mini batch data --> update PPO
        '''
        print('Step 12: Starting training')
        self.start_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)

        for self.conf.update in range(self.conf.update, self.conf.max_updates+1):
            # Parameter decay
            learning_rate, clip_range, entropy_coeff = get_decay(self.conf)

            # Sample training data
            sampled_episode_info, mean_rnd_reward, scaled_rewards = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            for b in self.buffer:
                b.prepare_batch_dict()

            # train K epochs
            actor_losses, critic_losses,  total_losses, dist_entropys, task_losses, rnd_losses = [[] for _ in range(6)]
            for _ in range(self.conf.K_epochs):
                mini_batch_generator = self._multi_buff_mini_batch_generator()
                for mini_batch in mini_batch_generator:
                    losses = self.ppo_agent.train_mini_batch(
                        learning_rate, clip_range, entropy_coeff, mini_batch, self.actual_sequence_length)
                    actor_losses.append(losses[0])
                    critic_losses.append(losses[1])
                    total_losses.append(losses[2])
                    dist_entropys.append(losses[3])
                    if losses[4] is not None:
                        task_losses.append(losses[4])
                    if losses[5] is not None:
                        rnd_losses.append(losses[5])

            # free memory
            [b.free_memory() for b in self.buffer]

            # write logs
            self.conf.i_episode += len(sampled_episode_info)
            episode_result = Trainer._process_episode_info(sampled_episode_info)
            self.logger.write_tensorboard((actor_losses, critic_losses, total_losses, dist_entropys, task_losses, rnd_losses,
                                          learning_rate, clip_range, entropy_coeff, mean_rnd_reward, episode_result, scaled_rewards), self.conf.update)
            self.logger.write_reward(self.conf.update, self.conf.i_episode, episode_result)

            # save model weights
            if self.conf.update != 0 and self.conf.update % self.conf.save_model_freq == 0:
                self._save()

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
        ''' reset all environment in workers '''
        if isinstance(self.obs_space, (gym_spaces.Tuple, gymnasium_spaces.Tuple)):
            obs = [[np.zeros(o.shape, dtype=np.float32) for o in self.obs_space] for _ in range(self.conf.num_workers)]
        else:
            obs = [[np.zeros(self.obs_space.shape, dtype=np.float32)] for _ in range(self.conf.num_workers)]

        # reset env
        for worker in self.workers:
            worker.child.send(('reset', None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            obs[w] = worker.child.recv()
        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        recurrent_cell = self.ppo_agent.init_recurrent_cell_states(self.conf.num_workers)
        # reset reward scaling
        if self.conf.use_reward_scaling:
            for rs in self.reward_scaling:
                rs.reset()
        return obs, recurrent_cell

    def _sample_training_data(self) -> list:
        '''Runs all n workers for n steps to sample training data.
        Returns:
            {list} -- list of results of completed episodes.
        '''
        episode_infos = []
        scaled_rewards = []
        episode_reward = [[] for _ in range(self.conf.num_workers)]
        # Sample actions from the model and collect experiences for training
        for t in range(self.conf.worker_steps):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Preprocess state data
                state_t = _obs_2_tensor(self.obs, self.conf.device)
                # gnenrate select mask
                if not hasattr(self, 'mask') or not hasattr(self, 'indices') or not hasattr(self, 'buffer_mask') or not hasattr(self, 'original_order'):
                    self.mask, self.worker_2_buff, self.wroker_2_buff_subw = self._state_classified_mask(state_t)
                    self.buffer_mask = torch.arange(0, self.conf.num_workers).reshape(
                        self.conf.task_num, self.conf.num_workers // self.conf.task_num).to(self.conf.device)
                    self.original_order = torch.argsort(torch.cat(self.mask))

                reordered_state_t = [torch.index_select(s, 0, torch.cat(self.mask)) for s in state_t]
                for i, m in enumerate(self.buffer_mask):
                    self.buffer[i].obs[t] = [torch.index_select(v, 0, m) for v in reordered_state_t]

                if self.conf.use_lstm:
                    for i, m in enumerate(self.buffer_mask):
                        if self.conf.layer_type == 'gru':
                            self.buffer[i].hxs[:, t] = torch.index_select(self.recurrent_cell.squeeze(0), 0, m)
                        elif self.conf.layer_type == 'lstm':
                            self.buffer[i].hxs[:, t] = torch.index_select(self.recurrent_cell[0].squeeze(0), 0, m)
                            self.buffer[i].cxs[:, t] = torch.index_select(self.recurrent_cell[1].squeeze(0), 0, m)

                # Forward the model
                action_t, action_logprob_t, value_t, rnd_value_t, self.recurrent_cell = self._multi_task_select_action(
                    reordered_state_t, self.buffer_mask)

                # save to diffrent buffer
                for i, m in enumerate(self.buffer_mask):
                    self.buffer[i].actions[:, t] = torch.index_select(action_t, 0, m)
                    self.buffer[i].log_probs[:, t] = torch.index_select(action_logprob_t, 0, m)
                    self.buffer[i].values[:, t] = torch.index_select(value_t, 0, m)
                    if self.conf.use_rnd:
                        self.buffer[i].rnd_values[:, t] = torch.index_select(rnd_value_t, 0, m)

            # restore action order
            restored_action_t = torch.index_select(action_t, 0, self.original_order)

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', restored_action_t[w].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs_w, reward_w, done_w, info = worker.child.recv()
                buffer_index = self.worker_2_buff[w]
                subworker_index = self.wroker_2_buff_subw[w]
                self.buffer[buffer_index].rewards[subworker_index, t] = self.reward_scaling[w](
                    reward_w) if self.conf.use_reward_scaling else reward_w
                self.buffer[buffer_index].dones[subworker_index, t] = done_w

                # debug
                episode_reward[w].append(self.buffer[buffer_index].rewards[subworker_index, t])

                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(('reset', None))
                    # Get data from reset
                    obs_w = worker.child.recv()
                    # Reset recurrent cell states
                    if self.conf.use_lstm and self.conf.reset_hidden_state:
                        rc = self.ppo_agent.init_recurrent_cell_states(1)
                        index = self.original_order[w].item()
                        if self.conf.layer_type == 'lstm':
                            self.recurrent_cell[0][:, index] = rc[0]
                            self.recurrent_cell[1][:, index] = rc[1]
                        else:
                            self.recurrent_cell[:, index] = rc

                    # reset reward scaling
                    if self.conf.use_reward_scaling:
                        self.reward_scaling[w].reset()

                    # debug
                    scaled_rewards.append(np.mean(episode_reward[w]))
                    episode_reward[w] = []

                # Store latest observations
                self.obs[w] = obs_w

            # save next obs in buffer for rnd
            if self.conf.use_rnd:
                _state_t = _obs_2_tensor(self.obs, self.conf.device)
                _rnd_state_t = torch.index_select(_state_t[0], 0, torch.cat(self.mask))
                for i, m in enumerate(self.buffer_mask):
                    self.buffer[i].rnd_next_obs[:, t] = torch.index_select(_rnd_state_t, 0, m)

        # Calculate internal reward
        if self.conf.use_rnd:
            next_state = torch.cat([b.rnd_next_obs for b in self.buffer], dim=0)
            total_rnd_rewards = self.ppo_agent.policy.rnd.calculate_rnd_rewards(next_state)

            if not hasattr(self, 'rnd_reward_index'):
                num_worker_task = [b.rnd_rewards.shape[0] for b in self.buffer]
                self.rnd_reward_index = np.cumsum(num_worker_task)
            rnd_rewards = np.split(total_rnd_rewards[:self.rnd_reward_index[-1]], self.rnd_reward_index[:-1])
            for b, r in zip(self.buffer, rnd_rewards):
                b.rnd_rewards = r
                # TODO 每个worker normalize ？
                b.normalize_rnd_rewards()
            mean_rnd_reward = total_rnd_rewards.mean().item()
        else:
            mean_rnd_reward = None

        # Calculate advantages
        state_t = _obs_2_tensor(self.obs, self.conf.device)
        state_t = [torch.index_select(s, 0, torch.cat(self.mask)) for s in state_t]
        _, _, last_value_t, last_rnd_value_t, _ = self._multi_task_select_action(state_t, self.buffer_mask)
        for b, m in zip(self.buffer, self.buffer_mask):
            if self.conf.use_rnd:
                b.calc_advantages(torch.index_select(last_value_t, 0, m), torch.index_select(last_rnd_value_t, 0, m))
            else:
                b.calc_advantages(torch.index_select(last_value_t, 0, m), None)
        return episode_infos, mean_rnd_reward, scaled_rewards

    def _multi_task_select_action(self, reshaped_state, buffer_mask):
        task_data = []
        for i, m in enumerate(buffer_mask):
            task_state = [torch.index_select(s, 0, m) for s in reshaped_state]
            if self.conf.use_lstm:
                task_hidden_in = torch.index_select(self.recurrent_cell, 1, m) if self.conf.layer_type == 'gru' else \
                    (torch.index_select(self.recurrent_cell[0], 1, m), torch.index_select(self.recurrent_cell[1], 1, m))

            task_data.append(self.ppo_agent.select_action(task_state, task_hidden_in, i))

        action_t, action_logprob_t, value_t, ext_value_t, recurrent_cell_t = zip(*task_data)
        action_t = torch.cat(action_t, dim=0)
        action_logprob_t = torch.cat(action_logprob_t, dim=0)
        value_t = torch.cat(value_t, dim=0)
        ext_value_t = torch.cat(ext_value_t, dim=0) if self.conf.use_rnd else None
        if self.conf.use_lstm:
            if self.conf.layer_type == 'gru':
                recurrent_cell_t = torch.cat(recurrent_cell_t, 1)
            elif self.conf.layer_type == 'lstm':
                recurrent_cell_t = (torch.cat([rc[0] for rc in recurrent_cell_t], dim=1),
                                    torch.cat([rc[1] for rc in recurrent_cell_t], dim=1))
        else:
            recurrent_cell_t = None
        return action_t, action_logprob_t, value_t, ext_value_t, recurrent_cell_t

    def _state_classified_mask(self, state: list):
        ''' According to the taskid in the state, classify the state and generate a classified mask.
        Args:
            state {list} -- state like: [tensor(feature), tensor(task_id)]

        Returns:
            {list} -- classified mask list like: [tensor(m1),tensor(m2),...]
            {list} -- sort indices
        '''
        if self.conf.multi_task:
            indices = np.argmax(state[-1].cpu().numpy(), axis=1)
            unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
            mask = [torch.tensor(np.where(inverse_indices == val)[0]).to(self.conf.device)
                    for val in unique_indices]
        else:
            mask = [torch.tensor(np.arange(self.conf.num_workers)).to(self.conf.device)]
            indices = [0 for _ in range(self.conf.num_workers)]
        reward_worker_map = {}
        for w in range(self.conf.num_workers):
            reward_worker_map[w] = torch.where(torch.stack(mask, dim=0) == w)[1].item()
        return mask, indices, reward_worker_map

    @ staticmethod
    def _process_episode_info(episode_info: list) -> dict:
        '''Extracts the mean and std of completed episode statistics like length and total reward.
        Args:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        '''
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == 'success':
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + '_percent'] = np.sum(episode_result) / len(episode_result)
                result[key + '_mean'] = np.mean([info[key] for info in episode_info])
                result[key + '_std'] = np.std([info[key] for info in episode_info])
        return result

    def close(self, done: bool = True):
        '''close traning.
        Args:
            done {bool} -- Whether the maximum number of training steps has been reached
        '''
        if not done:
            self._save()
        self.logger.close()

        for worker in self.workers:
            worker.child.send(('close', None))

        print('============================================================================================')
        end_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)
        print('Finished training at (GMT) : ', end_time)
        print('Total training time  : ', end_time - self.start_time)
        print('============================================================================================')

    def _save(self):
        '''save model & reward scaling & yaml file'''

        # save model
        print('--------------------------------------------------------------------------------------------')
        self.logger.save_checkpoint(self.conf.update, self.ppo_agent)

        # save reward scaling
        if self.conf.use_reward_scaling:
            self.logger.save_pickle(self.reward_scaling, 'reward_scaling.pkl')

        # save yaml file
        self.conf.save(self.logger.run_log_dir)
        print('--------------------------------------------------------------------------------------------')

    def _signal_handler(self, sig, frame):
        '''save when keyboard interrupt'''
        self.close(done=False)
