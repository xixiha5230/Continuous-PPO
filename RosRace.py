"""
沿着跑到避障
"""
import argparse

import numpy as np
from gym.spaces import Box, Tuple

from RosMain import RL_CAMERA, RL_LASER, RL_STATE, RosMain


class RosRace(RosMain):
    def _custom_rl_obs_space(self, *obs_shapes):
        obs_shapes = Tuple(
            [
                Box(low=0, high=255, shape=obs_shapes[0], dtype=np.uint8),
                Box(low=0, high=255, shape=obs_shapes[1], dtype=np.uint8),
                Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            ]
        )
        return obs_shapes

    def _custom_rl_action_space(self, action_shape):
        action_s = Box(low=-1, high=1, shape=(action_shape,), dtype=np.float32)
        return action_s

    def _custom_rl_obs_list(self, *obs_list):
        obs_list = list(obs_list)
        obs_list[2] = obs_list[2][..., :2]  # 只使用速度信息
        return obs_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config file")
    parser.add_argument("--ckpt", help="ckeckpoint to restore")
    args = parser.parse_args()

    RosRace(
        args.config_file,
        args.ckpt,
        rl_keys=[RL_CAMERA, RL_LASER, RL_STATE],
        decision_period=2,
        decision_rate=20,
        motor_limit=0.75,
    )
