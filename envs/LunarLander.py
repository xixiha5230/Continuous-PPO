import gym
from utils.normalization import RewardScaling


class LunarLander:
    def __init__(self, continuous=False, render_mode=None):
        self._env = gym.make(
            "LunarLander-v2", continuous=continuous, render_mode=render_mode)
        self.reward_scaling = RewardScaling(1, 0.99)
        self.reward_scaling.reset()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()
        self.reward_scaling.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._rewards.append(reward)
        done = terminated or truncated
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        return obs, self.reward_scaling(reward), done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
