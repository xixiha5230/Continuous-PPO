import gym
import numpy as np
import time


class CartPole:
    def __init__(self, name, render_mode=None):
        self._env = gym.make(name, render_mode=render_mode)

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._rewards.append(reward)
        done = terminated or truncated
        if done:
            info = {'reward': sum(self._rewards),
                    'length': len(self._rewards)}
        else:
            info = None
        return obs, reward / 100.0, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
