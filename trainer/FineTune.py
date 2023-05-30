from datetime import datetime

import numpy as np
import torch

from layers.TaskNet import TaskPredictNet
from trainer.Trainer import Trainer
from utils.polynomial_decay import get_decay


class FineTune(Trainer):
    def run(self):
        '''
        sample data --> prepare batch data --> gengerate mini batch data --> update PPO
        '''
        print('Step 12: Starting training')
        self.start_time = datetime.now().replace(microsecond=0)
        print('Started training at (GMT) : ', self.start_time)

        m = []
        m.append(self.ppo_agent.policy.actor)
        m.append(self.ppo_agent.policy.obs_net)
        m.append(self.ppo_agent.policy.task_net)
        m.append(self.ppo_agent.policy.critic)
        m.append(self.ppo_agent.policy.rnn_net)
        m.append(self.ppo_agent.policy.hidden_net)
        for model in m:
            for p in model.parameters():
                p.requires_grad = False

        self.ppo_agent.policy.task_predict_net = TaskPredictNet(
            self.conf.hidden_layer_size, 64, self.conf.task_num).to(self.conf.device)
        self.ppo_agent.policy.task_predict_net.requires_grad_(True)
        self.ppo_agent.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.ppo_agent.policy.parameters()), lr=self.conf.lr_schedule['final'], eps=1e-5)

        for self.conf.update in range(self.conf.update, self.conf.max_updates+self.conf.fine_tune_steps + 1):
            # Parameter decay
            learning_rate = self.conf.lr_schedule['final']
            clip_range = self.conf.clip_range_schedule['final']
            entropy_coeff = self.conf.entropy_coeff_schedule['final']

            # Sample training data
            sampled_episode_info = self._sample_training_data()

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

            # write logs
            self.conf.i_episode += len(sampled_episode_info)
            episode_result = Trainer._process_episode_info(sampled_episode_info)
            self.logger.write_tensorboard((actor_losses, critic_losses, total_losses, dist_entropys, task_losses, rnd_losses,
                                           learning_rate, clip_range, entropy_coeff,  episode_result,
                                           np.mean([b.rewards for b in self.buffer]),
                                           np.mean([b.rnd_rewards for b in self.buffer]) if self.conf.use_rnd else None),
                                          self.conf.update)
            self.logger.write_reward(self.conf.update, self.conf.i_episode, episode_result)

            # save model weights
            if self.conf.update != 0 and self.conf.update % self.conf.save_model_freq == 0:
                self._save()

            # free memory
            [b.free_memory() for b in self.buffer]
