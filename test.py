import glob
import os
import time
import argparse
import yaml
import imageio
import torch
from algorithm.PPO import PPO
from utils.env_helper import create_env

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='PPO_logs/Hallway/No_LSTM_seq32_256/run_0/config.yaml',
    help='The config file',
)
parser.add_argument(
    '--save_gif',
    type=bool,
    default=False
)


def test(args):
    if(torch.cuda.is_available()):
        device = 'cuda'
        torch.cuda.empty_cache()
        print('Device set to : ' + str(torch.cuda.get_device_name(device)))
    else:
        device = 'cpu'
        print('Device set to : cpu')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    conf_train = config['train']
    env_name = conf_train['env_name']
    action_type = conf_train['action_type']
    exp_name = conf_train['exp_name']
    run_num = conf_train['run_num']

    render = True
    frame_delay = 0.01
    total_test_episodes = 1 if args.save_gif else 10

    env = create_env(env_name, action_type=action_type,
                     render_mode='rgb_array' if args.save_gif else 'human', id=100, time_scale=1)
    observation_space = env.observation_space
    if action_type == 'continuous':
        action_space = env.action_space
    elif action_type == 'discrete':
        action_space = env.action_space.n
    else:
        raise NotImplementedError(action_type)

    # initialize a PPO agent
    ppo_agent = PPO(observation_space, action_space, config)

    # load latest checkpoint
    log_dir = f'./PPO_logs/{env_name}/{exp_name}/run_{run_num}'
    latest_checkpoint = max(glob.glob(f'{log_dir}/checkpoints/*'), key=os.path.getctime)
    # latest_checkpoint = f'{log_dir}/checkpoints/290.pth'
    print(f'resume from {latest_checkpoint}')
    ppo_agent.load(latest_checkpoint)

    # start testing
    test_running_reward = 0
    images = []
    for ep in range(1, total_test_episodes+1):
        state = env.reset()
        h_out = ppo_agent.init_recurrent_cell_states(1)
        while True:
            h_in = h_out
            action, _, _, _, _, h_out = ppo_agent.select_action(state, h_in)
            state, _, _, info = env.step(action[0])
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
    avg_test_reward = round(avg_test_reward, 2)
    print('average test reward : ' + str(avg_test_reward))
    print('============================================================================================')


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)
