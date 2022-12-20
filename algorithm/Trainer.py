import yaml
import signal
import os
import glob

import torch
import numpy as np
import tensorboardX
from algorithm.PPO import PPO

from datetime import datetime


class Trainer:
    def __init__(self, env, args) -> None:
        with open(args.config, 'r') as infile:
            self.config = yaml.safe_load(infile)
        signal.signal(signal.SIGINT, self._signal_handler)

        ####### initialize environment hyperparameters ######
        self.env_name = self.config['env_name']
        # continuous action space; else discrete
        self.has_continuous_action_space = self.config.get(
            'has_continuous_action_space', True)
        # max timesteps in one episode
        self.max_ep_len = self.config.get('max_ep_len', 1000)
        # break training loop if timeteps > max_training_timesteps
        self.max_training_timesteps = self.config.get(
            'max_training_timesteps', int(1e7))

        # print avg reward in the interval (in num timesteps)
        self.print_freq = self.config.get('print_freq', self.max_ep_len * 10)
        # log avg reward in the interval (in num timesteps)
        self.log_freq = self.config.get('log_freq', self.max_ep_len * 2)
        # save model frequency (in num timesteps)
        self.save_model_freq = self.config.get('save_model_freq', int(1e5))

        # starting std for action distribution (         )
        self.action_std = self.config.get('action_std', 0.6)
        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.action_std_decay_rate = self.config.get(
            'action_std_decay_rate', 0.05)
        # minimum action_std (stop decay after action_std <= min_action_std)
        self.min_action_std = self.config.get('min_action_std', 0.1)
        # action_std decay frequency (in num timesteps)
        self.action_std_decay_freq = self.config.get(
            'action_std_decay_freq', int(2.5e5))
        #####################################################

        # Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################
        # update policy every n timesteps
        self.update_timestep = self.config.get(
            'update_timestep', self.max_ep_len * 4)
        # update policy for K epochs in one PPO update
        self.K_epochs = self.config.get('K_epochs', 80)
        # clip parameter for PPO
        self.eps_clip = self.config.get('eps_clip', 0.2)
        # discount factor
        self.gamma = self.config.get('gamma', 0.99)
        # learning rate for actor network
        self.lr_actor = self.config.get('lr_actor', 0.0003)
        # learning rate for critic network
        self.lr_critic = self.config.get('lr_critic', 0.001)
        # set random seed if required (0 = no random seed)
        self.random_seed = self.config.get('random_seed', 0)
        #####################################################

        self.env = env
        # state space dimension
        self.state_dim = env.observation_space.shape[0]

        # action space dimension
        if self.has_continuous_action_space:
            self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = env.action_space.n

        ###################### logging ######################
        # log files for multiple runs are NOT overwritten
        self.log_dir = "PPO_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.log_dir = f'{self.log_dir}/{self.env_name}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # get number of log files in log directory
        current_num_files = next(os.walk(self.log_dir))[1]
        self.run_num = self.config.get('run_num', len(current_num_files))
        self.resume = self.config.get('resume', False)
        self.config['run_num'] = self.run_num

        # create new log file for each run
        self.log_dir = f'{self.log_dir}/run_{self.run_num}'
        if os.path.exists(self.log_dir) and not self.resume:
            raise FileExistsError(f"{self.log_dir} already exists!")
        self.log_f_name = f"{self.log_dir}/reward.csv"

        print("current logging run number for " +
              self.env_name + " : ", self.run_num)
        print("logging at : " + self.log_f_name)

        # tensorboard
        self.writer = tensorboardX.SummaryWriter(
            log_dir=self.log_dir)
        #####################################################

        ################### checkpointing ###################
        # change this to prevent overwriting weights in same env_name folder
        self.checkpoint_path = f'{self.log_dir}/checkpoints'
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        print("save checkpoint path : " + self.checkpoint_path)
        #####################################################
        self._print_all_hyp()

        ################# training procedure ################
        # initialize a PPO agent
        self.ppo_agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma,
                             self.K_epochs, self.eps_clip, self.has_continuous_action_space, self.action_std)
        if self.resume:
            latest_checkpoint = max(
                glob.glob(f'{self.log_dir}/checkpoints/*'), key=os.path.getctime)
            print(f"resume from {latest_checkpoint}")
            self.ppo_agent.load(latest_checkpoint)
        # track total training time
        self.start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)

        print("============================================================================================")

        # logging file
        self.log_f = open(self.log_f_name, "a+")
        if not self.resume:
            self.log_f.write('episode,timestep,reward\n')

        # printing and logging variables
        self.print_running_reward = self.config.get('print_running_reward', 0)
        self.print_running_episodes = self.config.get(
            'print_running_episodes', 0)

        self.log_running_reward = self.config.get('log_running_reward', 0)
        self.log_running_episodes = self.config.get('log_running_episodes', 0)

        self.time_step = self.config.get('time_step', 0)
        self.i_episode = self.config.get('i_episode', 0)

    def run(self):
        # training loop
        while self.time_step <= self.max_training_timesteps:

            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.max_ep_len+1):

                # select action with policy
                action = self.ppo_agent.select_action(state)
                state, reward, done, info = self.env.step(action)

                # saving reward and is_terminals
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                self.time_step += 1
                current_ep_reward += reward

                # update PPO agent
                if self.time_step % self.update_timestep == 0:
                    actor_loss, critic_loss, dist_entropy, total_loss = self.ppo_agent.update()
                    self.writer.add_scalar(
                        "Loss/actor", actor_loss, self.i_episode)
                    self.writer.add_scalar(
                        "Loss/critic", critic_loss, self.i_episode)
                    self.writer.add_scalar(
                        "Loss/total", total_loss, self.i_episode)
                    self.writer.add_scalar(
                        "entropy", dist_entropy, self.i_episode)

                # if continuous action space; then decay action std of ouput action distribution
                if self.has_continuous_action_space and self.time_step % self.action_std_decay_freq == 0:
                    self.ppo_agent.decay_action_std(
                        self.action_std_decay_rate, self.min_action_std)

                # log in logging file
                if self.time_step % self.log_freq == 0:
                    # log average reward till last episode
                    log_avg_reward = self.log_running_reward / self.log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)
                    self.log_f.write('{},{},{}\n'.format(
                        self.i_episode, self.time_step, log_avg_reward))
                    self.log_f.flush()
                    self.writer.add_scalar(
                        "reward", log_avg_reward, self.i_episode)
                    self.log_running_reward = 0
                    self.log_running_episodes = 0

                # printing average reward
                if self.time_step % self.print_freq == 0:
                    # print average reward till last episode
                    print_avg_reward = self.print_running_reward / self.print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        self.i_episode, self.time_step, print_avg_reward))

                    self.print_running_reward = 0
                    self.print_running_episodes = 0

                # save model weights
                if self.time_step % self.save_model_freq == 0:
                    self._save()
                # break; if the episode is over
                if done:
                    break

            self.print_running_reward += current_ep_reward
            self.print_running_episodes += 1

            self.log_running_reward += current_ep_reward
            self.log_running_episodes += 1

            self.i_episode += 1

        self.log_f.close()
        self.writer.close()
        self.env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", self.start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - self.start_time)
        print("============================================================================================")

    def _save(self):
        print(
            "--------------------------------------------------------------------------------------------")
        checkpoint_file = f"{self.checkpoint_path}/{self.i_episode}.pth"
        print("saving model at : " + checkpoint_file)
        self.ppo_agent.save(checkpoint_file)
        print("model saved")
        print("Elapsed Time  : ", datetime.now().replace(
            microsecond=0) - self.start_time)
        self.config['print_running_reward'] = float(self.print_running_reward)
        self.config['print_running_episodes'] = self.print_running_episodes
        self.config['log_running_reward'] = float(self.log_running_reward)
        self.config['log_running_episodes'] = self.log_running_episodes
        self.config['time_step'] = self.time_step
        self.config['i_episode'] = self.i_episode
        self.config['action_std'] = self.ppo_agent.action_std
        self.config['resume'] = True
        yaml_file = f'{self.log_dir}/config.yaml'
        print(f"save configures at: {yaml_file}")
        with open(yaml_file, 'w') as fp:
            yaml.dump(self.config, fp)
        print(
            "--------------------------------------------------------------------------------------------")

    def _print_all_hyp(self):
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", self.max_training_timesteps)
        print("max timesteps per episode : ", self.max_ep_len)
        print("model saving frequency : " +
              str(self.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " +
              str(self.print_freq) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", self.state_dim)
        print("action space dimension : ", self.action_dim)
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            print("Initializing a continuous action space policy")
            print(
                "--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", self.action_std)
            print("decay rate of std of action distribution : ",
                  self.action_std_decay_rate)
            print("minimum std of action distribution : ", self.min_action_std)
            print("decay frequency of std of action distribution : " +
                  str(self.action_std_decay_freq) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " +
              str(self.update_timestep) + " timesteps")
        print("PPO K epochs : ", self.K_epochs)
        print("PPO epsilon clip : ", self.eps_clip)
        print("discount factor (gamma) : ", self.gamma)
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)
        if self.random_seed:
            print(
                "--------------------------------------------------------------------------------------------")
            print("setting random seed to ", self.random_seed)
            torch.manual_seed(self.random_seed)
            self.env.seed(self.random_seed)
            np.random.seed(self.random_seed)
        print("============================================================================================")

    def _signal_handler(self, sig, frame):
        self._save()
        self.log_f.close()
        self.writer.close()
        self.env.close()
        exit(0)
