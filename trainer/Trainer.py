import yaml
import signal
import os
import glob
import pickle

import torch
import numpy as np
import tensorboardX
from datetime import datetime

from algorithm.PPO import PPO
from utils.env_helper import create_env
from replaybuffer.Buffer import Buffer
from worker.Worker import Worker
from gymnasium import spaces
from utils.polynomial_decay import polynomial_decay
from normalization.RewardScaling import RewardScaling


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

        print('Step 3: Obtain the action space and observation space')
        dummy_env = create_env(self.env_name, self.action_type)
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
        self.buffer = Buffer(self.conf, self.obs_space, self.action_space)

        print('Step 6: Init model and optimizer')
        self.ppo_agent = PPO(self.obs_space, self.action_space, self.conf)

        print('Step 7: Init environment workers')
        self.workers = [Worker(self.env_name, self.action_type, w) for w in range(self.num_workers)]

        print('Step 8: Random seed')
        if self.random_seed != 0:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

        print('Step 9: Reset workers')
        self.obs, self.recurrent_cell = self._reset_env()

        print('Step 10: Set log files')
        self.log_dir = f'PPO_logs/{self.env_name}/{self.exp_name}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        num_files = next(os.walk(self.log_dir))[1]
        self.run_num = self.conf_train.setdefault('run_num', len(num_files))
        self.log_dir = f'{self.log_dir}/run_{self.run_num}'
        log_f_name = f'{self.log_dir}/reward.csv'

        print('Step 11: Check resume')
        if not self.resume:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            else:
                raise FileExistsError(f'{self.log_dir} already exists!')
            self.log_f = open(log_f_name, 'a+')
            self.log_f.write('update,\tepisode,\treward\n')
        else:
            self.log_f = open(log_f_name, 'a+')
            latest_checkpoint = max(glob.glob(f'{self.log_dir}/checkpoints/*'), key=os.path.getctime)
            print(f'resume from {latest_checkpoint}')
            self.ppo_agent.load(latest_checkpoint)
            if self.use_reward_scaling:
                reward_scaling_file = f'{self.log_dir}/reward_scaling.pkl'
                with open(reward_scaling_file, 'rb') as f:
                    self.reward_scaling = pickle.load(f)
        self.writer = tensorboardX.SummaryWriter(log_dir=self.log_dir)
        self.checkpoint_path = f'{self.log_dir}/checkpoints'
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        print('save checkpoint path : ' + self.checkpoint_path)

    def run(self):
        '''
        sample data --> prepare batch data --> gengerate mini batch data --> update PPO
        '''
        print('Step 12: Starting training')
        self.start_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)

        for self.update in range(self.max_updates):
            # Parameter decay
            learning_rate = polynomial_decay(self.lr_schedule['init'], self.lr_schedule['final'],
                                             self.lr_schedule['max_decay_steps'], self.lr_schedule['pow'], self.update)
            clip_range = polynomial_decay(self.clip_range_schedule['init'], self.clip_range_schedule['final'],
                                          self.clip_range_schedule['max_decay_steps'], self.clip_range_schedule['pow'], self.update)
            entropy_coeff = polynomial_decay(self.entropy_coeff_schedule['init'], self.entropy_coeff_schedule['final'],
                                             self.entropy_coeff_schedule['max_decay_steps'], self.entropy_coeff_schedule['pow'], self.update)

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            # train K epochs
            for _ in range(self.K_epochs):
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
                actor_loss_mean = []
                critic_loss_mean = []
                dist_entropy_mean = []
                total_loss_mean = []
                for mini_batch in mini_batch_generator:
                    actor_loss, critic_loss, loss, dist_entropy = self.ppo_agent._train_mini_batch(
                        learning_rate, clip_range, entropy_coeff, mini_batch, self.buffer.actual_sequence_length)
                    actor_loss_mean.append(actor_loss)
                    critic_loss_mean.append(critic_loss)
                    dist_entropy_mean.append(dist_entropy)
                    total_loss_mean.append(loss)

            # update old policy
            # self.ppo_agent.old_policy.load_state_dict(self.ppo_agent.policy.state_dict())

            # free memory
            self.buffer.free_memory()

            # write logs
            self.i_episode += len(sampled_episode_info)
            episode_result = Trainer._process_episode_info(sampled_episode_info)
            self.writer.add_scalar('Loss/actor', np.mean(actor_loss_mean), self.update)
            self.writer.add_scalar('Loss/critic', np.mean(critic_loss_mean), self.update)
            self.writer.add_scalar('Loss/total', np.mean(total_loss_mean), self.update)
            self.writer.add_scalar('Loss/entropy', np.mean(dist_entropy_mean), self.update)
            self.writer.add_scalar('Parameter/learning_rate', learning_rate, self.update)
            self.writer.add_scalar('Parameter/clip_range', clip_range, self.update)
            self.writer.add_scalar('Parameter/entropy_coeff', entropy_coeff, self.update)
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

    def _reset_env(self):
        ''' reset all environment in workers '''
        if isinstance(self.obs_space, spaces.Tuple):
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
                state_t = PPO._state_2_tensor(self.obs, self.device)
                if isinstance(state_t, list):
                    self.buffer.obs[t] = state_t
                else:
                    self.buffer.obs[:, t] = state_t
                if self.use_lstm:
                    if self.layer_type == 'gru':
                        self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                    elif self.layer_type == 'lstm':
                        self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0)
                        self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model
                action_t, action_logprob_t, value_t, self.recurrent_cell = self.ppo_agent.select_action(
                    state_t, self.recurrent_cell)

                self.buffer.actions[:, t] = action_t
                self.buffer.log_probs[:, t] = action_logprob_t
                self.buffer.values[:, t] = value_t
            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(('step', action_t[w].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs_w, reward_w, done_w, info = worker.child.recv()
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
                        if self.layer_type == 'lstm':
                            rc = self.ppo_agent.init_recurrent_cell_states(1)
                            self.recurrent_cell[0][:, w] = rc[0]
                            self.recurrent_cell[1][:, w] = rc[1]
                        else:
                            self.recurrent_cell[:, w] = self.ppo_agent.init_recurrent_cell_states(1)
                    # reset reward scaling
                    if self.use_reward_scaling:
                        self.reward_scaling[w].reset()
                # Store latest observations
                self.obs[w] = obs_w

        # Calculate advantages
        _, _, last_value_t, _ = self.ppo_agent.select_action(PPO._state_2_tensor(self.obs, self.device), self.recurrent_cell)
        self.buffer.calc_advantages(last_value_t)
        return episode_infos

    @staticmethod
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
        for worker in self.workers:
            worker.child.send(('close', None))
        print('============================================================================================')
        end_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)
        print('Finished training at (GMT) : ', end_time)
        print('Total training time  : ', end_time - self.start_time)
        print('============================================================================================')
        exit(0)

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
        checkpoint_file = f'{self.checkpoint_path}/{self.update}.pth'
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
        self.close(done=False)
