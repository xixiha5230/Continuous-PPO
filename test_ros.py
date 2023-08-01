import argparse
from algorithm.PPO import PPO
from gym.spaces import Box, Tuple as TupleSpace
import numpy as np
from ros_car import RosCar, LASER_SCAN_SIZE
import time
import yaml
import os
import glob
from utils.Logger import Logger
from utils.ConfigHelper import ConfigHelper
from utils.obs_2_tensor import _obs_2_tensor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_file",
    type=str,
    default="PPO_logs/xx.yaml",
    help="The config file",
)
parser.add_argument(
    "--mode",
    type=str,
    default="car_race",
    help="The car mode",
)
parser.add_argument(
    "--ckpt",
    type=int,
    default=0,
    help="The checkpoint index",
)
parser.add_argument(
    "--speed",
    type=float,
    default=0.3,
    help="Car run speed.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    speed = args.speed
    custom_ckpt = args.ckpt
    config = ConfigHelper(args.config_file)
    logger = Logger(config.env_name, config.exp_name, config.run_num, True, True)

    # init ros car
    car = RosCar()
    while not car.initialized():
        time.sleep(0.5)
    print("car initialized")
    print("laser_scan_size", len(car.laser_scan))
    if LASER_SCAN_SIZE != len(car.laser_scan):
        print(f"warning! laser_scan_size != {LASER_SCAN_SIZE}")
    # car.init_ros_plt()

    # 参数加载
    obs_table = {
        "car_race": TupleSpace(
            [Box(0.0, 1.0, (84, 84, 3)), Box(float("-inf"), float("inf"), (400,))]
        ),
    }
    observation_space = obs_table["car_race"]
    action_space = Box(-1.0, 1.0, (2,))
    state_normalizer = logger.load_pickle("state_normalizer.pkl")

    # PPO初始化
    ppo_agent = PPO(observation_space, action_space, config)
    latest_checkpoint = max(
        glob.glob(os.path.join(logger.checkpoint_path, "*")), key=os.path.getctime
    )
    if custom_ckpt != 0:
        latest_checkpoint = f"{logger.checkpoint_path}/{custom_ckpt}.pth"
    print(f"resume from {latest_checkpoint}")
    ppo_agent.load(latest_checkpoint)
    ppo_agent.policy.eval()
    h_out = ppo_agent.init_recurrent_cell_states(1)
    while not car.is_shutdown():
        # car.update_ros_plt()
        start = time.time()
        state = car.get_rl_obs_list()
        state = state_normalizer(state, update=False)
        action, h_out, task_predict = ppo_agent.eval_select_action(
            _obs_2_tensor(state, config.device), h_out
        )
        action = action.cpu().numpy()[0]

        car.move_car(action[0] * speed, -action[1])
        # print(action[0] * speed, action[1])

        end = time.time()
        print(f"time: {(end - start)*1000}ms")
