import glob
import os
import pickle

import numpy as np
import tensorboardX


class Logger:
    def __init__(self, env_name, exp_name, run_num, resume, test=False) -> None:
        base_log_dir = os.path.join("PPO_logs", env_name, exp_name)
        os.makedirs(base_log_dir, exist_ok=True)
        self.run_num = len(next(os.walk(base_log_dir))[1]) if not resume else run_num
        self.run_log_dir = os.path.join(base_log_dir, f"run_{self.run_num}")
        if not resume:
            os.makedirs(self.run_log_dir)

        reward_file = os.path.join(self.run_log_dir, "reward.csv")
        if not resume:
            self.reward_writter = open(reward_file, "w+")
            self.reward_writter.write("update,\tepisode,\treward\n")
        else:
            self.reward_writter = open(reward_file, "a+")
        if not test:
            self.writer = tensorboardX.SummaryWriter(log_dir=self.run_log_dir)
        self.checkpoint_path = os.path.join(self.run_log_dir, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        if not test:
            print("save checkpoint path : " + self.checkpoint_path)

    def write_reward(self, step, i_episode, episode_result):
        if len(episode_result) > 0:
            self.reward_writter.write(
                f'{step},\t{i_episode},\t{episode_result["reward_mean"]}\n'
            )
            self.reward_writter.flush()
            print(
                f'step: {step}\t episode: {i_episode}\t reward: {episode_result["reward_mean"]}'
            )

    @property
    def latest_checkpoint(self):
        latest_checkpoint = max(
            glob.glob(os.path.join(self.run_log_dir, "checkpoints", "*")),
            key=os.path.getctime,
        )
        print(f"resume from {latest_checkpoint}")
        return latest_checkpoint

    def save_checkpoint(self, update, agent):
        checkpoint_file = os.path.join(self.checkpoint_path, f"{update}.pth")
        agent.save(checkpoint_file)
        print("saving model at : " + checkpoint_file)

    def load_pickle(self, name):
        file = os.path.join(self.run_log_dir, name)
        assert os.path.exists(file) == True, f"{file} not found"
        with open(file, "rb") as f:
            return pickle.load(f)

    def save_pickle(self, obj, name):
        file = os.path.join(self.run_log_dir, name)
        with open(file, "wb") as f:
            pickle.dump(obj, f)

    def write_tensorboard(self, datas, step):
        (
            actor_loss_mean,
            critic_loss_mean,
            total_loss_mean,
            dist_entropy_mean,
            task_loss_mean,
            rnd_loss_mean,
            learning_rate,
            clip_range,
            entropy_coeff,
            task_coeff,
            episode_result,
            scaled_rewards,
            mean_rnd_reward,
        ) = datas

        self.writer.add_scalar("Loss/actor", np.mean(actor_loss_mean), step)
        self.writer.add_scalar("Loss/critic", np.mean(critic_loss_mean), step)
        self.writer.add_scalar("Loss/total", np.mean(total_loss_mean), step)
        self.writer.add_scalar("Loss/entropy", np.mean(dist_entropy_mean), step)
        if len(task_loss_mean) != 0:
            self.writer.add_scalar("Loss/task", np.mean(task_loss_mean), step)
        if len(rnd_loss_mean) != 0:
            self.writer.add_scalar("Loss/rnd", np.mean(rnd_loss_mean), step)
        self.writer.add_scalar("Parameter/learning_rate", learning_rate, step)
        self.writer.add_scalar("Parameter/clip_range", clip_range, step)
        self.writer.add_scalar("Parameter/entropy_coeff", entropy_coeff, step)
        self.writer.add_scalar("Parameter/task_coeff", task_coeff, step)
        if mean_rnd_reward != None:
            self.writer.add_scalar("Train/rnd_reward_mean", mean_rnd_reward, step)
        if len(episode_result) > 0:
            self.writer.add_scalar(
                "Train/reward_mean", episode_result["reward_mean"], step
            )
            self.writer.add_scalar("Train/scaled_reward", scaled_rewards, step)
            self.writer.add_scalar(
                "Train/reward_std", episode_result["reward_std"], step
            )
            self.writer.add_scalar(
                "Train/length_mean", episode_result["length_mean"], step
            )
            self.writer.add_scalar(
                "Train/length_std", episode_result["length_std"], step
            )

    def close(self):
        self.reward_writter.close()
        self.writer.close()
