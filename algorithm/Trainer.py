import yaml
import signal
import os
import glob

import torch
import numpy as np
import tensorboardX
from datetime import datetime

from algorithm.PPO import PPO
from utils.env_helper import create_env
from replaybuffer.Buffer import Buffer
from worker.Worker import Worker
from gym import spaces

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = 'cpu'
if(torch.cuda.is_available()):
    device = 'cuda'
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class Trainer:
    def __init__(self, args) -> None:
        # load config
        self.args = args
        self.conf = {}
        with open(args.config, 'r') as infile:
            self.conf = yaml.safe_load(infile)
        self._config_check()
        # handle signal
        signal.signal(signal.SIGINT, self._signal_handler)

        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.env_name, self.has_continuous_action_space)
        self.obs_space = dummy_env.observation_space
        if self.has_continuous_action_space:
            self.action_dim = dummy_env.action_space.shape[0]
        else:
            self.action_dim = dummy_env.action_space.n
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.conf, self.obs_space, self.action_dim)

        # Init a PPO agent
        print("Step 3: Init model and optimizer")
        self.ppo_agent = PPO(self.obs_space,
                             self.action_dim, self.conf)

        # Init workers
        print("Step 4: Init environment workers")
        num_workers = self.conf_worker['num_workers']
        self.workers = [Worker(
            self.env_name, self.has_continuous_action_space, w) for w in range(num_workers)]

        if self.random_seed != 0:
            # TODO 设置seed会导致效果不好
            torch.manual_seed(self.random_seed)
            # self.env.seed(self.random_seed)
            np.random.seed(self.random_seed)

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        self.obs, self.recurrent_cell = self._reset_env()

        # log files for multiple runs are NOT overwritten
        print("Step 6: Set logs")
        self.log_dir = f'PPO_logs/{self.env_name}/{self.exp_name}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # get number of log files in log directory
        current_num_files = next(os.walk(self.log_dir))[1]
        self.run_num = self.conf.setdefault(
            'run_num', len(current_num_files))
        self.log_dir = f'{self.log_dir}/run_{self.run_num}'
        log_f_name = f"{self.log_dir}/reward.csv"
        print("logging at : " + log_f_name)
        if not self.resume:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            else:
                raise FileExistsError(f"{self.log_dir} already exists!")
            self.log_f = open(log_f_name, "a+")
            self.log_f.write('update,\tepisode,\treward\n')
        else:
            self.log_f = open(log_f_name, "a+")
            latest_checkpoint = max(
                glob.glob(f'{self.log_dir}/checkpoints/*'), key=os.path.getctime)
            print(f"resume from {latest_checkpoint}")
            self.ppo_agent.load(latest_checkpoint)

        # tensorboard
        self.writer = tensorboardX.SummaryWriter(
            log_dir=self.log_dir)
        # change this to prevent overwriting weights in same env_name folder
        self.checkpoint_path = f'{self.log_dir}/checkpoints'
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        print("save checkpoint path : " + self.checkpoint_path)

    def run(self):
        # training loop
        print("Step 7: Starting training")
        self.start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)

        for self.update in range(self.update, self.max_updates+1):

            # Sample training data
            sampled_episode_info = self._sample_training_data()

            # Prepare the sampled data inside the buffer (splits data into sequences)
            self.buffer.prepare_batch_dict()

            for _ in range(self.K_epochs):
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
                actor_loss_mean = []
                critic_loss_mean = []
                dist_entropy_mean = []
                total_loss_mean = []
                for mini_batch in mini_batch_generator:
                    actor_loss, critic_loss, loss, dist_entropy = self.ppo_agent._train_mini_batch(
                        mini_batch)
                    actor_loss_mean.append(actor_loss)
                    critic_loss_mean.append(critic_loss)
                    dist_entropy_mean.append(
                        dist_entropy)
                    total_loss_mean.append(loss)
            self.ppo_agent.policy_old.load_state_dict(
                self.ppo_agent.policy.state_dict())

            self.i_episode += len(sampled_episode_info)
            self.writer.add_scalar(
                "Loss/actor", np.mean(actor_loss_mean), self.i_episode)
            self.writer.add_scalar(
                "Loss/critic", np.mean(critic_loss_mean), self.i_episode)
            self.writer.add_scalar(
                "Loss/total", np.mean(total_loss_mean), self.i_episode)
            self.writer.add_scalar(
                "entropy", np.mean(dist_entropy_mean), self.i_episode)
            self.writer.add_scalar(
                "reward", np.mean(sampled_episode_info), self.i_episode)
            print(
                f'update: {self.update}\t episode: {self.i_episode}\t reward: {np.mean(sampled_episode_info)}')
            # write to file
            self.log_f.write(
                f'{self.update},\t{self.i_episode},\t{np.mean(sampled_episode_info)}\n')
            self.log_f.flush()
            # save model weights
            if self.update % self.save_model_freq == 0:
                self._save()

    def _reset_env(self):
        num_workers = self.conf_worker['num_workers']
        if isinstance(self.obs_space, spaces.Tuple):
            obs = [[np.zeros(t.shape, dtype=np.float32)
                    for t in self.obs_space] for _ in range(num_workers)]
        else:
            obs = np.zeros(
                (num_workers,) + self.obs_space.shape, dtype=np.float32)
        # reset env
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            obs[w] = worker.child.recv()
        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        recurrent_cell = self.ppo_agent.init_recurrent_cell_states(num_workers)
        return obs, recurrent_cell

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.
        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        episode_start = np.zeros(
            (self.conf_worker['num_workers']), dtype=np.int64)
        
        # Sample actions from the model and collect experiences for training
        for t in range(self.conf_worker['worker_steps']):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():

                if self.conf_recurrence['use_lstm']:
                    if self.conf_recurrence["layer_type"] == "gru":
                        self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                    elif self.conf_recurrence["layer_type"] == "lstm":
                        self.buffer.hxs[:,
                                        t] = self.recurrent_cell[0].squeeze(0)
                        self.buffer.cxs[:,
                                        t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model
                action, state_t, action_t, action_logprob_t, self.recurrent_cell = self.ppo_agent.select_action(
                    self.obs, self.recurrent_cell)

                if isinstance(state_t, list):
                    self.buffer.obs[t] = state_t
                else:
                    self.buffer.obs[:, t] = state_t
                self.buffer.actions[:, t] = action_t
                self.buffer.log_probs[:, t] = action_logprob_t

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", action[w]))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs_w, reward_w, done_w, info = worker.child.recv()
                self.buffer.rewards[w, t] = reward_w
                self.buffer.dones[w, t] = done_w
                if done_w:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(
                        self.buffer.rewards[w, episode_start[w]:t+1].sum())
                    episode_start[w] = t+1
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs_w = worker.child.recv()
                    # Reset recurrent cell states
                    if self.conf_recurrence['use_lstm'] and self.conf_recurrence["reset_hidden_state"]:
                        self.recurrent_cell[:, w] = self.ppo_agent.init_recurrent_cell_states(
                            1)
                # Store latest observations
                self.obs[w] = obs_w

        # Calculate advantages
        self.buffer.calc_advantages(self.gamma)
        return episode_infos

    def close(self):
        self._save()
        self.writer.close()
        for worker in self.workers:
            worker.child.send(("close", None))
        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - self.start_time)
        print("============================================================================================")
        exit(0)

    def _config_check(self):
        ####### initialize environment hyperparameters ######
        self.env_name = self.conf['env_name']
        self.exp_name = self.conf.setdefault('exp_name', self.args.exp_name)
        # max updates times
        self.max_updates = self.conf.setdefault('max_updates', 150)
        self.update = self.conf.setdefault('update', 1)
        self.i_episode = self.conf.setdefault('i_episode', 1)
        # save model frequency (in num update)
        self.save_model_freq = self.conf.setdefault('save_model_freq', 5)
        self.resume = self.conf.setdefault('resume', False)
        self.device = self.conf.setdefault('device', device)
        self.random_seed = self.conf.setdefault('random_seed', 0)

        ################ PPO hyperparameters ################
        conf_ppo = {}
        self.conf_ppo = self.conf.setdefault('ppo', conf_ppo)
        self.vf_loss_coeff = self.conf_ppo.setdefault('vf_loss_coeff', 0.5)
        self.entropy_coeff = self.conf_ppo.setdefault('entropy_coeff', 0.001)
        # update policy for K epochs in one PPO update
        self.K_epochs = self.conf.setdefault('K_epochs', 80)
        self.num_mini_batch = self.conf.setdefault('num_mini_batch', 4)
        # clip parameter for PPO
        self.eps_clip = self.conf.setdefault('eps_clip', 0.2)
        # discount factor
        self.gamma = self.conf.setdefault('gamma', 0.99)
        # learning rate for actor network
        self.lr_actor = self.conf.setdefault('lr_actor', 0.0003)
        # learning rate for critic network
        self.lr_critic = self.conf.setdefault('lr_critic', 0.001)
        # continuous action space; else discrete
        self.has_continuous_action_space = self.conf.setdefault(
            'has_continuous_action_space', True)

        ############## LSTM hyperparameters #################
        recurrence = {}
        self.conf_recurrence = self.conf.setdefault('recurrence', recurrence)
        self.conf_recurrence.setdefault('use_lstm', False)
        self.conf_recurrence.setdefault('sequence_length', 8)
        self.conf_recurrence.setdefault('hidden_state_size', 64)
        self.conf_recurrence.setdefault('layer_type', 'gru')
        self.conf_recurrence.setdefault('reset_hidden_state', True)

        ############## Worker hyperparameters #################
        worker = {}
        self.conf_worker = self.conf.setdefault('worker', worker)
        self.conf_worker.setdefault('num_workers', 6)
        self.conf_worker.setdefault('worker_steps', 1000)

    def _save(self):
        print(
            "--------------------------------------------------------------------------------------------")
        checkpoint_file = f"{self.checkpoint_path}/{self.i_episode}.pth"
        print("saving model at : " + checkpoint_file)
        self.ppo_agent.save(checkpoint_file)

        # args need uodate
        self.conf['update'] = self.update
        self.conf['i_episode'] = self.i_episode
        self.conf['resume'] = True

        yaml_file = f'{self.log_dir}/config.yaml'
        print(f"save configures at: {yaml_file}")
        with open(yaml_file, 'w') as fp:
            yaml.dump(self.conf, fp)
        print(
            "--------------------------------------------------------------------------------------------")

    def _signal_handler(self, sig, frame):
        self.close()
