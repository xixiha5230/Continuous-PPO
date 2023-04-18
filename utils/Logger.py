import glob
import os
import pickle

import numpy as np
import tensorboardX


class Logger:
    def __init__(self, env_name, exp_name, run_num, resume) -> None:
        base_log_dir = os.path.join('PPO_logs', env_name, exp_name)
        os.makedirs(base_log_dir, exist_ok=True)
        run_num = len(next(os.walk(base_log_dir))[1]) if not resume else run_num
        self.run_log_dir = os.path.join(base_log_dir, f'run_{run_num}')
        os.makedirs(self.run_log_dir, exist_ok=True)

        reward_file = os.path.join(self.run_log_dir, 'reward.csv')
        if not resume:
            self.reward_writter = open(reward_file, 'w+')
            self.reward_writter.write('update,\tepisode,\treward\n')
        else:
            self.reward_writter = open(reward_file, 'a+')

        self.writer = tensorboardX.SummaryWriter(log_dir=self.run_log_dir)
        self.checkpoint_path = os.path.join(self.run_log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        print('save checkpoint path : ' + self.checkpoint_path)

    def write_reward(self, x):
        self.reward_writter.write(x)
        self.reward_writter.flush()

    @property
    def latest_checkpoint(self):
        latest_checkpoint = max(glob.glob(os.path.join(self.run_log_dir, 'checkpoints', '*')), key=os.path.getctime)
        print(f'resume from {latest_checkpoint}')
        return latest_checkpoint

    def save_checkpoint(self, update, agent):
        checkpoint_file = os.path.join(self.checkpoint_path, f'{update}.pth')
        agent.save(checkpoint_file)
        print('saving model at : ' + checkpoint_file)

    def load_pickle(self, name):
        reward_scaling_file = os.path.join(self.run_log_dir, name)
        with open(reward_scaling_file, 'rb') as f:
            return pickle.load(f)

    def save_pickle(self, obj, name):
        reward_scaling_file = os.path.join(self.run_log_dir, name)
        with open(reward_scaling_file, 'wb') as f:
            pickle.dump(obj, f)

    def write_tensorboard(self, loss, update):
        (actor_loss_mean,
         critic_loss_mean,
         total_loss_mean,
         dist_entropy_mean,
         task_loss_mean,
         rnd_loss_mean,
         learning_rate,
         clip_range,
         entropy_coeff,
         mean_rnd_reward,
         episode_result,
         scaled_rewards) = loss

        self.writer.add_scalar('Loss/actor', np.mean(actor_loss_mean), update)
        self.writer.add_scalar('Loss/critic', np.mean(critic_loss_mean), update)
        self.writer.add_scalar('Loss/total', np.mean(total_loss_mean), update)
        self.writer.add_scalar('Loss/entropy', np.mean(dist_entropy_mean), update)
        if len(task_loss_mean) != 0:
            self.writer.add_scalar('Loss/task', np.mean(task_loss_mean), update)
        if len(rnd_loss_mean) != 0:
            self.writer.add_scalar('Loss/rnd', np.mean(rnd_loss_mean), update)
        self.writer.add_scalar('Parameter/learning_rate', learning_rate, update)
        self.writer.add_scalar('Parameter/clip_range', clip_range, update)
        self.writer.add_scalar('Parameter/entropy_coeff', entropy_coeff, update)
        if mean_rnd_reward != None:
            self.writer.add_scalar('Train/rnd_reward_mean', mean_rnd_reward, update)
        if(len(episode_result) > 0):
            self.writer.add_scalar('Train/reward_mean', episode_result['reward_mean'], update)
            self.writer.add_scalar('Train/scaled_reward', np.mean(scaled_rewards), update)
            self.writer.add_scalar('Train/reward_std', episode_result['reward_std'], update)
            self.writer.add_scalar('Train/length_mean', episode_result['length_mean'], update)
            self.writer.add_scalar('Train/length_std', episode_result['length_std'], update)

    def close(self):
        self.reward_writter.close()
        self.writer.close()
