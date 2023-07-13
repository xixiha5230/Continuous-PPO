from typing import Optional

import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv


class HalfCheetahBace(HalfCheetahEnv):
    def __init__(
        self,
        forward_reward_weight=1,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        super().__init__(
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )
        self._count_step = 0
        self._max_step = 4000
        self._rewards = []

    def _get_obs(self):
        return (
            np.concatenate(
                [
                    self.data.qpos.flat[1:],
                    self.data.qvel.flat,
                    self.get_body_com("torso").flat,
                ]
            )
            .astype(np.float32)
            .flatten()
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._count_step = 0
        self._rewards = []
        super().reset(seed=seed, options=options)
        obs = self._get_obs()
        return obs

    def step(self, action):
        obs, reward, _, _ = self._step(action)
        self._rewards.append(reward)
        self._count_step += self.frame_skip
        if self._count_step >= self._max_step:
            if self._count_step > self._max_step + self.frame_skip:
                raise UserWarning("need rest after done!")
            done = True
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            done = False
            info = None
        return obs, reward, done, info

    def _step(self, action):
        raise NotImplementedError
