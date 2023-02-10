import gym
import numpy as np
import time


class CarRacing:
    def __init__(self, continuous, render_mode='human'):
        self._env = gym.make(
            "CarRacing-v2", domain_randomize=False, continuous=continuous, render_mode='human')

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()
        obs = obs / 255.0
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = obs / 255.0
        self._rewards.append(reward)
        done = terminated or truncated
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs, reward / 100.0, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
