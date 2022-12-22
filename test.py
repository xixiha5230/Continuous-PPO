import glob
import os
import time
import argparse
import yaml
import imageio
import gym
from algorithm.PPO import PPO

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    # default="configs/LunarLander-v2.yaml",
    default="PPO_logs/LunarLander-v2/run_1/config.yaml",
    help="The config file",
)


def test(args):
    with open(args.config, 'r') as infile:
        config = yaml.safe_load(infile)
    print("============================================================================================")
    env_name = config['env_name']
    has_continuous_action_space = config['has_continuous_action_space']

    max_ep_len = 1000           # max timesteps in one episode
    # set same std for action distribution which was used while saving
    action_std = 0.1
    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames
    total_test_episodes = 10    # total num of testing episodes

    env = gym.make(env_name, continuous=True)
    # state space dimension
    state_dim = env.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, config)
    # set to 0.1 when test
    ppo_agent.action_std = action_std
    log_dir = f"./PPO_logs/{env_name}/run_{config['run_num']}"
    latest_checkpoint = max(
        glob.glob(f'{log_dir}/checkpoints/*'), key=os.path.getctime)
    latest_checkpoint = f'{log_dir}/checkpoints/30211.pth'
    print(f"resume from {latest_checkpoint}")
    ppo_agent.load(latest_checkpoint)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    images = []
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward

            if render:
                env.render(mode="human")
                images.append(env.render(mode='rgb_array'))
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
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
