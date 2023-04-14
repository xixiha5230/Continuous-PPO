import glob
import os
import pickle
import signal
from datetime import datetime

import numpy as np
import tensorboardX
import torch
import yaml
from gym import spaces as gym_spaces
from gymnasium import spaces as gymnasium_spaces

from algorithm.PPO import PPO
from normalization.RewardScaling import RewardScaling
from replaybuffer.Buffer import Buffer
from utils.env_helper import create_env
from utils.obs_2_tensor import _obs_2_tensor
from utils.polynomial_decay import polynomial_decay
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

    def __init__(self, config_file: str, exp_name: str) -> None:
        '''
        Args:
            config_file {str} -- path of yaml file that save the parameters
            exp_name {str} -- name of current experiment
        '''
        print('Step 1: Read the config file and checking')
        self.exp_name = exp_name
        self.conf = {}
        with open(config_file, 'r') as infile:
            self.conf = yaml.safe_load(infile)
        self._config_check()

        print('Step 2: Listen for keyboard interrupts to save the model')
        signal.signal(signal.SIGINT, self._signal_handler)
        self.stop_signal = False

        print('Step 3: Obtain the action space and observation space')
        dummy_env = create_env(self.conf)
        self.obs_space = dummy_env.observation_space
        if self.action_type == 'continuous':
            self.action_space = dummy_env.action_space
        elif self.action_type == 'discrete':
            self.action_space = dummy_env.action_space.n
        else:
            raise NotImplementedError(self.action_type)
        dummy_env.close()

        print('Step 4: Init reward scaling')
        if self.use_reward_scaling:
            self.reward_scaling = [RewardScaling(1, 0.99) for _ in range(self.num_workers)]
        else:
            print("don't use reward scaling")

        print('Step 5: Init buffer')
        if self.multi_task:
            self.task_num = len(self.conf.get('task', []))
            self.buffer = [Buffer(self.conf, self.obs_space, self.action_space) for _ in range(self.task_num)]
        else:
            self.buffer = Buffer(self.conf, self.obs_space, self.action_space)

        print('Step 6: Init model and optimizer')
        self.ppo_agent = PPO(self.obs_space, self.action_space, self.conf)

        print('Step 7: Init environment workers')
        self.workers = [Worker(self.conf, w) for w in range(self.num_workers)]

        print('Step 8: Random seed')
        if self.random_seed != 0:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        print('Step 9: Reset workers')
        self.obs, self.recurrent_cell = self._reset_env()

        print('Step 10: Set log files')
        self.log_dir = os.path.join('PPO_logs', self.env_name, self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        num_files = next(os.walk(self.log_dir))[1]
        self.run_num = self.conf_train.setdefault('run_num', len(num_files))
        self.log_dir = os.path.join(self.log_dir, f'run_{self.run_num}')
        log_f_name = os.path.join(self.log_dir, 'reward.csv')

        print('Step 11: Check resume')
        if not self.resume:
            if os.path.exists(self.log_dir):
                raise FileExistsError(f'{self.log_dir} already exists!')
            os.makedirs(self.log_dir)
            self.log_f = open(log_f_name, 'a+')
            self.log_f.write('update,\tepisode,\treward\n')
        else:
            self.log_f = open(log_f_name, 'a+')
            latest_checkpoint = max(glob.glob(os.path.join(self.log_dir, 'checkpoints', '*')), key=os.path.getctime)
            print(f'resume from {latest_checkpoint}')
            self.ppo_agent.load(latest_checkpoint)
            if self.use_reward_scaling:
                reward_scaling_file = os.path.join(self.log_dir, 'reward_scaling.pkl')
                with open(reward_scaling_file, 'rb') as f:
                    self.reward_scaling = pickle.load(f)
        self.writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)
        self.checkpoint_path = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        print('save checkpoint path : ' + self.checkpoint_path)

    def run(self):
        '''
        sample data --> prepare batch data --> gengerate mini batch data --> update PPO
        '''
        print('Step 12: Starting training')
        self.start_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)

        for self.update in range(self.update, self.max_updates):
            # Parameter decay
            learning_rate = polynomial_decay(self.lr_schedule['init'], self.lr_schedule['final'],
                                             self.lr_schedule['max_decay_steps'], self.lr_schedule['pow'], self.update)
            clip_range = polynomial_decay(self.clip_range_schedule['init'], self.clip_range_schedule['final'],
                                          self.clip_range_schedule['max_decay_steps'], self.clip_range_schedule['pow'], self.update)
            entropy_coeff = polynomial_decay(self.entropy_coeff_schedule['init'], self.entropy_coeff_schedule['final'],
                                             self.entropy_coeff_schedule['max_decay_steps'], self.entropy_coeff_schedule['pow'], self.update)

            # Sample training data
            try:
                sampled_episode_info, mean_rnd_reward = self._sample_training_data()
            except Exception as e:
                if self.stop_signal:
                    break
                else:
                    raise e

            # Prepare the sampled data inside the buffer (splits data into sequences)
            if self.multi_task:
                for b in self.buffer:
                    b.prepare_batch_dict()
            else:
                self.buffer.prepare_batch_dict()

            # train K epochs
            for _ in range(self.K_epochs):
                if self.multi_task:
                    mini_batch_generator = self._multi_buff_mini_batch_generator(
                        [b.recurrent_mini_batch_generator() for b in self.buffer])
                    actual_sequence_length = self.buffer[0].actual_sequence_length
                else:
                    mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
                    actual_sequence_length = self.buffer.actual_sequence_length

                actor_loss_mean = []
                critic_loss_mean = []
                dist_entropy_mean = []
                total_loss_mean = []
                task_loss_mean = []
                rnd_loss_mean = []
                for mini_batch in mini_batch_generator:
                    actor_loss, critic_loss, loss, dist_entropy, task_loss, rnd_loss = self.ppo_agent.train_mini_batch(
                        learning_rate, clip_range, entropy_coeff, mini_batch, actual_sequence_length)
                    actor_loss_mean.append(actor_loss)
                    critic_loss_mean.append(critic_loss)
                    dist_entropy_mean.append(dist_entropy)
                    total_loss_mean.append(loss)
                    if task_loss is not None:
                        task_loss_mean.append(task_loss)
                    if rnd_loss is not None:
                        rnd_loss_mean.append(rnd_loss)
            # free memory
            if self.multi_task:
                for b in self.buffer:
                    b.free_memory()
            else:
                self.buffer.free_memory()

            # write logs
            self.i_episode += len(sampled_episode_info)
            episode_result = Trainer._process_episode_info(sampled_episode_info)
            self.writer.add_scalar('Loss/actor', np.mean(actor_loss_mean), self.update)
            self.writer.add_scalar('Loss/critic', np.mean(critic_loss_mean), self.update)
            self.writer.add_scalar('Loss/total', np.mean(total_loss_mean), self.update)
            self.writer.add_scalar('Loss/entropy', np.mean(dist_entropy_mean), self.update)
            if len(task_loss_mean) != 0:
                self.writer.add_scalar('Loss/task', np.mean(task_loss_mean), self.update)
            if len(rnd_loss_mean) != 0:
                self.writer.add_scalar('Loss/rnd', np.mean(rnd_loss_mean), self.update)
            self.writer.add_scalar('Parameter/learning_rate', learning_rate, self.update)
            self.writer.add_scalar('Parameter/clip_range', clip_range, self.update)
            self.writer.add_scalar('Parameter/entropy_coeff', entropy_coeff, self.update)
            if mean_rnd_reward != None:
                self.writer.add_scalar('Train/rnd_reward_mean', mean_rnd_reward, self.update)
            if(len(episode_result) > 0):
                self.writer.add_scalar('Train/reward_mean', episode_result['reward_mean'], self.update)
                self.writer.add_scalar('Train/reward_std', episode_result['reward_std'], self.update)
                self.writer.add_scalar('Train/length_mean', episode_result['length_mean'], self.update)
                self.writer.add_scalar('Train/length_std', episode_result['length_std'], self.update)
                print(f'update: {self.update}\t episode: {self.i_episode}\t reward: {episode_result["reward_mean"]}')
                self.log_f.write(f'{self.update},\t{self.i_episode},\t{episode_result["reward_mean"]}\n')
                self.log_f.flush()

            # save model weights
            if self.update != 0 and self.update % self.save_model_freq == 0:
                self._save()

    def _multi_buff_mini_batch_generator(self, mini_batch_generators: list):
        while True:
            try:
                mini_batch = [next(mg) for mg in mini_batch_generators]
                yield mini_batch
            except StopIteration:
                break

    def _reset_env(self):
        ''' reset all environment in workers '''
        if isinstance(self.obs_space, (gym_spaces.Tuple, gymnasium_spaces.Tuple)):
            obs = [[np.zeros(t.shape, dtype=np.float32) for t in self.obs_space] for _ in range(self.num_workers)]
        else:
            obs = np.zeros((self.num_workers,) + self.obs_space.shape, dtype=np.float32)
        # reset env
        for worker in self.workers:
            worker.child.send(('reset', None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            obs[w] = worker.child.recv()
        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        recurrent_cell = self.ppo_agent.init_recurrent_cell_states(self.num_workers)
        # reset reward scaling
        if self.use_reward_scaling:
            for rs in self.reward_scaling:
                rs.reset()
        return obs, recurrent_cell

    def _sample_training_data(self) -> list:
        '''Runs all n workers for n steps to sample training data.
        Returns:
            {list} -- list of results of completed episodes.
        '''
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.worker_steps):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Preprocess state data
                state_t = _obs_2_tensor(self.obs, self.device)
                if isinstance(state_t, list):
                    if self.multi_task:
                        # gnenrate select mask
                        if not hasattr(self, 'mask') or not hasattr(self, 'indices') or not hasattr(self, 'buffer_mask') or not hasattr(self, 'original_order'):
                            self.mask, self.indices = self._state_classified_mask(state_t)
                            self.buffer_mask = torch.arange(0, self.num_workers).reshape(
                                self.task_num, self.num_workers // self.task_num).to(self.device)
                            self.original_order = torch.argsort(torch.cat(self.mask))
                        reshaped_state_t = [torch.index_select(s, 0, torch.cat(self.mask)) for s in state_t]
                        for i, m in enumerate(self.buffer_mask):
                            self.buffer[i].obs[t] = [torch.index_select(v, 0, m) for v in reshaped_state_t]
                    else:
                        self.buffer.obs[t] = state_t
                else:
                    assert self.multi_task == False
                    self.buffer.obs[:, t] = state_t
                if self.use_lstm:
                    if self.layer_type == 'gru':
                        # save to diffrent buffer,self.recurrent_cell already shaped
                        if self.multi_task:
                            for i, m in enumerate(self.buffer_mask):
                                self.buffer[i].hxs[:, t] = torch.index_select(self.recurrent_cell.squeeze(0), 0, m)
                        else:
                            self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                    elif self.layer_type == 'lstm':
                        # save to diffrent buffer,self.recurrent_cell already shaped
                        if self.multi_task:
                            for i, m in enumerate(self.buffer_mask):
                                self.buffer[i].hxs[:, t] = torch.index_select(self.recurrent_cell[0].squeeze(0), 0, m)
                                self.buffer[i].hxs[:, t] = torch.index_select(self.recurrent_cell[1].squeeze(0), 0, m)
                        else:
                            self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0)
                            self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model
                if self.multi_task:
                    action_t, action_logprob_t, value_t, rnd_value_t, self.recurrent_cell = self._multi_task_select_action(
                        reshaped_state_t, self.buffer_mask)
                else:
                    action_t, action_logprob_t, value_t, rnd_value_t, self.recurrent_cell = self.ppo_agent.select_action(
                        state_t, self.recurrent_cell)
                if self.multi_task:
                    # save to diffrent buffer
                    for i, m in enumerate(self.buffer_mask):
                        self.buffer[i].actions[:, t] = torch.index_select(action_t, 0, m)
                        self.buffer[i].log_probs[:, t] = torch.index_select(action_logprob_t, 0, m)
                        self.buffer[i].values[:, t] = torch.index_select(value_t, 0, m)
                        self.buffer[i].rnd_values[:, t] = torch.index_select(rnd_value_t, 0, m)
                else:
                    self.buffer.actions[:, t] = action_t
                    self.buffer.log_probs[:, t] = action_logprob_t
                    self.buffer.values[:, t] = value_t
                    self.buffer.rnd_values[:, t] = rnd_value_t
            # restore action order
            if self.multi_task:
                restored_action_t = torch.index_select(action_t, 0, self.original_order)
            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', restored_action_t[w].cpu().numpy()
                                  if self.multi_task else action_t[w].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs_w, reward_w, done_w, info = worker.child.recv()
                if self.multi_task:
                    self.buffer[self.indices[w]].rewards[w // self.task_num, t] = self.reward_scaling[w](
                        reward_w) if self.use_reward_scaling else reward_w
                    self.buffer[self.indices[w]].dones[w // self.task_num, t] = done_w
                else:
                    self.buffer.rewards[w, t] = self.reward_scaling[w](reward_w) if self.use_reward_scaling else reward_w
                    self.buffer.dones[w, t] = done_w
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(('reset', None))
                    # Get data from reset
                    obs_w = worker.child.recv()
                    # Reset recurrent cell states
                    if self.use_lstm and self.reset_hidden_state:
                        rc = self.ppo_agent.init_recurrent_cell_states(1)
                        if self.multi_task:
                            index = torch.where(torch.cat(self.mask) == w)[0].item()
                            if self.layer_type == 'lstm':
                                self.recurrent_cell[0][:, index] = rc[0]
                                self.recurrent_cell[1][:, index] = rc[1]
                            else:
                                self.recurrent_cell[:, w] = rc
                        else:
                            if self.layer_type == 'lstm':
                                self.recurrent_cell[0][:, w] = rc[0]
                                self.recurrent_cell[1][:, w] = rc[1]
                            else:
                                self.recurrent_cell[:, w] = rc
                    # reset reward scaling
                    if self.use_reward_scaling:
                        self.reward_scaling[w].reset()
                # Store latest observations
                self.obs[w] = obs_w

            # save next obs in buffer for rnd
            if self.use_rnd:
                _state_t = _obs_2_tensor(self.obs, self.device)
                if self.multi_task:
                    _rnd_state_t = torch.index_select(_state_t[0], 0, torch.cat(self.mask))
                    for i, m in enumerate(self.buffer_mask):
                        self.buffer[i].rnd_next_obs[:, t] = torch.index_select(_rnd_state_t, 0, m)
                else:
                    self.buffer.rnd_next_obs[:, t] = _state_t[0] if isinstance(_state_t, list) else _state_t

        # Calculate internal reward
        if self.use_rnd:
            next_state = torch.cat([b.rnd_next_obs for b in self.buffer],
                                   dim=0) if self.multi_task else self.buffer.rnd_next_obs
            total_rnd_rewards = self.ppo_agent.policy.rnd.calculate_rnd_rewards(next_state)
            if self.multi_task:
                num_worker_task = [b.rnd_rewards.shape[0] for b in self.buffer]
                idx = np.cumsum(num_worker_task)
                rnd_rewards = np.split(total_rnd_rewards[:idx[-1]], idx[:-1])
                for b, r in zip(self.buffer, rnd_rewards):
                    b.rnd_rewards = r
                    b.normalize_rnd_rewards()
            else:
                self.buffer.rnd_rewards = total_rnd_rewards
                self.buffer.normalize_rnd_rewards()
            mean_rnd_reward = total_rnd_rewards.mean().item()
        else:
            mean_rnd_reward = None

        # Calculate advantages
        state_t = _obs_2_tensor(self.obs, self.device)
        if self.multi_task:
            state_t = [torch.index_select(s, 0, torch.cat(self.mask)) for s in state_t]
            _, _, last_value_t, last_rnd_value_t, _ = self._multi_task_select_action(state_t, self.buffer_mask)
            for b, m in zip(self.buffer, self.buffer_mask):
                b.calc_advantages(torch.index_select(last_value_t, 0, m), torch.index_select(last_rnd_value_t, 0, m))
        else:
            _, _, last_value_t, last_rnd_value_t, _ = self.ppo_agent.select_action(state_t, self.recurrent_cell)
            self.buffer.calc_advantages(last_value_t, last_rnd_value_t)
        return episode_infos, mean_rnd_reward

    def _multi_task_select_action(self, reshaped_state, buffer_mask):
        action_t = None
        action_logprob_t = None
        value_t = None
        ext_value_t = None
        recurrent_cell_t = None
        for i, m in enumerate(buffer_mask):
            task_state = [torch.index_select(s, 0, m) for s in reshaped_state]
            if self.use_lstm:
                if self.layer_type == 'gru':
                    task_hidden_in = torch.index_select(self.recurrent_cell, 1, m)
                elif self.layer_type == 'lstm':
                    task_hidden_in = (torch.index_select(self.recurrent_cell[0], 1, m),
                                      torch.index_select(self.recurrent_cell[1], 1, m))
                else:
                    raise NotImplementedError(self.layer_type)
            action_t_, action_logprob_t_, value_t_, ext_value_t_, recurrent_cell_t_ = self.ppo_agent.select_action(
                task_state, task_hidden_in, i)
            action_t = torch.cat((action_t, action_t_), 0) if action_t is not None else action_t_
            action_logprob_t = torch.cat((action_logprob_t, action_logprob_t_),
                                         0) if action_logprob_t is not None else action_logprob_t_
            value_t = torch.cat((value_t, value_t_), 0) if value_t is not None else value_t_
            ext_value_t = torch.cat((ext_value_t, ext_value_t_), 0) if ext_value_t is not None else ext_value_t_
            if self.use_lstm:
                if self.layer_type == 'gru':
                    recurrent_cell_t = torch.cat((recurrent_cell_t, recurrent_cell_t_),
                                                 1) if recurrent_cell_t is not None else recurrent_cell_t_
                elif self.layer_type == 'lstm':
                    recurrent_cell_t = (torch.cat((recurrent_cell_t[0], recurrent_cell_t_[0]), 1),
                                        torch.cat((recurrent_cell_t[1], recurrent_cell_t_[1]), 1)) if recurrent_cell_t_ is not None else recurrent_cell_t_
                else:
                    raise NotImplementedError(self.layer_type)
        return action_t, action_logprob_t, value_t, ext_value_t, recurrent_cell_t

    def _state_classified_mask(self, state: list):
        ''' According to the taskid in the state, classify the state and generate a classified mask.
        Args:
            state {list} -- state like: [tensor(feature), tensor(task_id)]

        Returns:
            {list} -- classified mask list like: [tensor(m1),tensor(m2),...]
            {list} -- sort indices
        '''
        indices = np.argmax(state[-1].cpu().numpy(), axis=1)
        unique_indices, inverse_indices = np.unique(indices, return_inverse=True)
        mask = [torch.tensor(np.where(inverse_indices == val)[0]).to(self.device)
                for val in unique_indices]
        return mask, indices

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
        self.writer.close()
        try:
            for worker in self.workers:
                worker.child.send(('close', None))
        except BrokenPipeError as e:
            if self.stop_signal:
                pass
            else:
                raise e
        print('============================================================================================')
        end_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)
        print('Finished training at (GMT) : ', end_time)
        print('Total training time  : ', end_time - self.start_time)
        print('============================================================================================')

    def _config_check(self):
        '''check configuration file'''
        # check device
        print('============================================================================================')
        if(torch.cuda.is_available()):
            device = 'cuda'
            torch.cuda.empty_cache()
            print('Device set to : ' + str(torch.cuda.get_device_name(device)))
        else:
            device = 'cpu'
            print('Device set to : cpu')
        print('============================================================================================')

        # initialize train hyperparameters
        conf_train = {}
        self.conf_train: dict = self.conf.setdefault('train', conf_train)
        self.exp_name = self.conf_train.setdefault('exp_name', self.exp_name)
        self.env_name = self.conf_train['env_name']
        self.K_epochs = self.conf_train.setdefault('K_epochs', 80)
        self.device = self.conf_train.setdefault('device', device)
        self.action_type = self.conf_train.setdefault('action_type', 'continuous')
        self.save_model_freq = self.conf_train.setdefault('save_model_freq', 5)
        self.random_seed = self.conf_train.setdefault('random_seed', 0)
        self.use_reward_scaling = self.conf_train.setdefault('use_reward_scaling', True)
        self.max_updates = self.conf_train.setdefault('max_updates', 150)
        self.num_mini_batch = self.conf_train.setdefault('num_mini_batch', 4)
        self.hidden_layer_size = self.conf_train.setdefault('hidden_layer_size', 256)
        self.update = self.conf_train.setdefault('update', 0)
        self.i_episode = self.conf_train.setdefault('i_episode', 0)
        self.resume = self.conf_train.setdefault('resume', False)
        self.multi_task = self.conf_train.setdefault('multi_task', False)
        self.use_rnd = self.conf_train.setdefault('use_rnd', False)
        self.rnd_rate = self.conf_train.setdefault('rnd_rate', 0.5)

        # PPO hyperparameters
        conf_ppo = {}
        self.conf_ppo: dict = self.conf.setdefault('ppo', conf_ppo)
        self.gamma = self.conf_ppo.setdefault('gamma', 0.99)
        self.lamda = self.conf_ppo.setdefault('lamda', 0.95)
        self.vf_loss_coeff = self.conf_ppo.setdefault('vf_loss_coeff', 0.5)
        entropy_coeff_schedule = {}
        self.entropy_coeff_schedule: dict = self.conf_ppo.setdefault('entropy_coeff_schedule', entropy_coeff_schedule)
        self.entropy_coeff_schedule.setdefault('init', 0.001)
        self.entropy_coeff_schedule.setdefault('final', 0.001)
        self.entropy_coeff_schedule.setdefault('pow', 1.0)
        self.entropy_coeff_schedule.setdefault('max_decay_steps', 0)
        lr_schedule = {}
        self.lr_schedule: dict = self.conf_ppo.setdefault('lr_schedule', lr_schedule)
        self.lr_schedule.setdefault('init', 3.0e-4)
        self.lr_schedule.setdefault('final', 3.0e-4)
        self.lr_schedule.setdefault('pow', 1.0)
        self.lr_schedule.setdefault('max_decay_steps', 0)
        clip_range_schedule = {}
        self.clip_range_schedule: dict = self.conf_ppo.setdefault('clip_range_schedule', clip_range_schedule)
        self.clip_range_schedule.setdefault('init', 0.2)
        self.clip_range_schedule.setdefault('final', 0.2)
        self.clip_range_schedule.setdefault('pow', 1.0)
        self.clip_range_schedule.setdefault('max_decay_steps', 0)

        # LSTM hyperparameters
        recurrence = {}
        self.conf_recurrence: dict = self.conf.setdefault('recurrence', recurrence)
        self.use_lstm = self.conf_recurrence.setdefault('use_lstm', False)
        self.sequence_length = self.conf_recurrence.setdefault('sequence_length', -1)
        self.hidden_state_size = self.conf_recurrence.setdefault('hidden_state_size', 64)
        self.layer_type = self.conf_recurrence.setdefault('layer_type', 'gru')
        self.reset_hidden_state = self.conf_recurrence.setdefault('reset_hidden_state', True)

        # Worker hyperparameters
        worker = {}
        self.conf_worker = self.conf.setdefault('worker', worker)
        self.num_workers = self.conf_worker.setdefault('num_workers', 6)
        self.worker_steps = self.conf_worker.setdefault('worker_steps', 1000)

    def _save(self):
        '''save model & reward scaling & yaml file'''

        # save model
        print('--------------------------------------------------------------------------------------------')
        checkpoint_file = os.path.join(self.checkpoint_path, f'{self.update}.pth')
        print('saving model at : ' + checkpoint_file)
        self.ppo_agent.save(checkpoint_file)

        # save reward scaling
        if self.use_reward_scaling:
            reward_scaling_file = f'{self.log_dir}/reward_scaling.pkl'
            with open(reward_scaling_file, 'wb') as f:
                pickle.dump(self.reward_scaling, f)

        # save yaml file
        self.conf_train['update'] = self.update
        self.conf_train['i_episode'] = self.i_episode
        self.conf_train['resume'] = True
        yaml_file = f'{self.log_dir}/config.yaml'
        print(f'save configures at: {yaml_file}')
        with open(yaml_file, 'w') as fp:
            yaml.dump(self.conf, fp)
        print('--------------------------------------------------------------------------------------------')

    def _signal_handler(self, sig, frame):
        '''save when keyboard interrupt'''
        self.stop_signal = True
        self.close(done=False)
