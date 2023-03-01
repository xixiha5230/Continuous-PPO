import numpy as np
import gymnasium
from gymnasium.spaces import Tuple, Box, Discrete
from minigrid.wrappers import RGBImgPartialObsWrapper, ViewSizeWrapper


class Minigrid:
    ''' Gym Minigrid environment '''

    def __init__(self, env_name, render_mode=None):
        '''
        Args:
            name {str} -- name of diffrent Minigrid version
            render_mode {str} -- render mode: humnan or rgb_array
        '''
        self._old_env = gymnasium.make(env_name, render_mode=render_mode, agent_view_size=3)
        self._env = RGBImgPartialObsWrapper(self._old_env, 28)
        self._observation_space = Box(0, 1.0, (84, 84, 3))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return Discrete(3)

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset(seed=np.random.randint(0, 99))
        obs = obs["image"] / 255.0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = obs["image"] / 255.0
        self._rewards.append(reward)
        done = terminated or truncated
        if done:
            info = {'reward': sum(self._rewards),
                    'length': len(self._rewards)}
        else:
            info = None
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


if __name__ == '__main__':
    import cv2
    import time

    cv2.namedWindow("obs", 0)
    cv2.resizeWindow("obs", 300, 300)
    env = Minigrid('MiniGrid-MemoryS17Random-v0', render_mode='human')
    done = True
    while True:
        if done:
            obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        cv2.imshow('obs', obs)
        cv2.waitKey(1)
        time.sleep(1)
