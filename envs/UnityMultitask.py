import os
import sys

import numpy as np
from gym.spaces import Box, Tuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel

from utils.ConfigHelper import ConfigHelper


class UnityMultitask:
    """Unity multi task environment"""

    def __init__(self, config: ConfigHelper, worker_id: int, time_scale: int):
        """
        Args:
            task {dict} -- task config
            worker_id {int} -- unity work id
            time_scale {int} -- unnity time scale
        """
        task_num = config.task_num

        self.file_name = (
            config.env_win_path[worker_id // config.worker_per_task].get(
                "file_name", ""
            )
            if sys.platform == "win32"
            else config.env_linux_path[worker_id // config.worker_per_task].get(
                "file_name", ""
            )
        )
        self.file_name = os.path.join(config.glob_dir, self.file_name)
        self._one_hot = (
            config.env_win_path[worker_id // config.worker_per_task].get(
                "one_hot", [0] * task_num
            )
            if sys.platform == "win32"
            else config.env_linux_path[worker_id // config.worker_per_task].get(
                "one_hot", [0] * task_num
            )
        )
        self._observation_space = Tuple(
            [
                Box(float(-1), float(1), (84, 84, 3), np.float32),
                Box(float("-inf"), float("inf"), (400,), np.float32),
                Box(0, 1, (len(self._one_hot),), np.float32),
            ]
        )

        self.worker_id = worker_id
        self.channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(
            file_name=self.file_name, worker_id=worker_id, side_channels=[self.channel]
        )
        self._env = UnityToGymWrapper(
            unity_env, uint8_visual=False, allow_multiple_obs=True
        )
        self.channel.set_configuration_parameters(
            width=512,
            height=384,
            quality_level=5,
            time_scale=time_scale,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        obs.append(self._one_hot)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs.append(self._one_hot)
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


if __name__ == "__main__":
    import cv2

    task = [
        {"one_hot": [1, 0, 0, 0], "file_name": "UnityEnvs/UnityMultitask/Task0"},
        {"one_hot": [0, 1, 0, 0], "file_name": "UnityEnvs/UnityMultitask/Task1"},
    ]

    env = [UnityMultitask(task, id, time_scale=1) for id in range(2)]
    for i, e in enumerate(env):
        e.reset()
    while True:
        for i, e in enumerate(env):
            obs, reward, done, info = e.step(e.action_space.sample())
            cv2.imshow(f"Video{i}", cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            if info:
                e.reset()
