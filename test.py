import glob
import os
import time
import argparse
import yaml
import imageio
import pygame
import torch
import numpy as np
from algorithm.PPO import PPO
from utils.env_helper import create_env

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="PPO_logs/CarRace/lstm_continue/run_0/config.yaml",
    help="The config file",
)
parser.add_argument(
    "--save_gif",
    action='store_true'
)


def test(args):
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

    with open(args.config, 'r') as infile:
        config = yaml.safe_load(infile)
    print("============================================================================================")

    conf_train = config['train']
    env_name = conf_train['env_name']
    has_continuous_action_space = conf_train['has_continuous_action_space']
    exp_name = conf_train['exp_name']
    run_num = conf_train['run_num']

    max_ep_len = 2000           # max timesteps in one episode
    render = True              # render environment on screen
    frame_delay = 0.01             # if required; add delay b/w frames
    total_test_episodes = 10    # total num of testing episodes

    env = create_env(
        env_name, continuous=has_continuous_action_space, render_mode='human', id=100, time_scale=1)
    observation_space = env.observation_space
    if has_continuous_action_space:
        action_space = env.action_space
    else:
        action_space = env.action_space.n
    # initialize a PPO agent
    ppo_agent = PPO(observation_space, action_space, config)

    log_dir = f"./PPO_logs/{env_name}/{exp_name}/run_{run_num}"
    latest_checkpoint = max(
        glob.glob(f'{log_dir}/checkpoints/*'), key=os.path.getctime)
    # latest_checkpoint = f'{log_dir}/checkpoints/290.pth'
    print(f"resume from {latest_checkpoint}")
    ppo_agent.load(latest_checkpoint)
    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    images = []
    for ep in range(1, total_test_episodes+1):
        state = env.reset()
        h_out = ppo_agent.init_recurrent_cell_states(1)
        for t in range(1, max_ep_len+1):
            h_in = h_out
            action, _, _, _, _, h_out = ppo_agent.select_action(
                state, h_in)
            # action = ppo_agent.cal_action(np.array([state]), h_in)
            state, _, _, info = env.step(action[0])

            if render:
                env.render()
                # pygame.event.get()
                if args.save_gif:
                    images.append(env.render(mode='rgb_array'))
                time.sleep(frame_delay)

            if info:
                print(f'Episode: {ep} \t\t Reward: {info["reward"]}')
                test_running_reward += info["reward"]
                break

    if args.save_gif:
        imageio.mimsave(f'{log_dir}/test.gif', images)
    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
