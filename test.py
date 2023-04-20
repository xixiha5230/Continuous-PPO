import argparse
import glob
import os
import time

import imageio
import torch
import yaml

from algorithm.PPO import PPO
from utils.ConfigHelper import ConfigHelper
from utils.env_helper import create_env
from utils.obs_2_tensor import _obs_2_tensor

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='PPO_logs/MountainCar-v0/rnd/run_4/config.yaml',
    help='The config file',
)
parser.add_argument(
    '--save_gif',
    type=bool,
    default=False
)


def test(args):
    config = ConfigHelper(args.config)
    
    render = True
    frame_delay = 0.01
    total_test_episodes = 1 if args.save_gif else 10

    env = create_env(config, render_mode='rgb_array' if args.save_gif else 'human', id=101, time_scale=1)
    observation_space = env.observation_space
    if config.action_type == 'continuous':
        action_space = env.action_space
    elif config.action_type == 'discrete':
        action_space = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(observation_space, action_space, config)
    ppo_agent.policy.eval()

    # load latest checkpoint
    log_dir = f'./PPO_logs/{config.env_name}/{config.exp_name}/run_{config.run_num}'
    latest_checkpoint = max(glob.glob(f'{log_dir}/checkpoints/*'), key=os.path.getctime)
    # latest_checkpoint = f'{log_dir}/checkpoints/230.pth'
    print(f'resume from {latest_checkpoint}')
    ppo_agent.load(latest_checkpoint)

    # start testing
    test_running_reward = 0
    images = []
    with torch.no_grad():
        for ep in range(1, total_test_episodes+1):
            state = env.reset()
            h_out = ppo_agent.init_recurrent_cell_states(1)
            while True:
                h_in = h_out
                action, h_out = ppo_agent.eval_select_action(_obs_2_tensor(state, device), h_in)
                state, _, _, info = env.step(action[0].cpu().numpy())
                if render:
                    env.render()
                    # pygame.event.get()
                    if args.save_gif:
                        images.append(env.render())
                    time.sleep(frame_delay)
                if info:
                    print(f'Episode: {ep} \t\t Reward: {info["reward"]}')
                    test_running_reward += info['reward']
                    break

    if args.save_gif:
        imageio.mimsave(f'{log_dir}/test.gif', images)
    env.close()

    print('============================================================================================')
    avg_test_reward = test_running_reward / total_test_episodes
    print('average test reward : ' + str(avg_test_reward))
    print('============================================================================================')


if __name__ == '__main__':
    test(parser.parse_args())
