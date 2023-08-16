import gymnasium


class AdroitHandHammer:
    """Gym AdroitHandHammer environment"""

    def __init__(self, render_mode=None):
        """
        Args:
            render_mode {str} -- render mode: humnan or rgb_array
        """
        self._env = gymnasium.make(
            "AdroitHandHammer-v1", max_episode_steps=400, render_mode=render_mode
        )

    @property
    def observation_space(self):
        return gymnasium.spaces.Tuple((self._env.observation_space,))

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs, _ = self._env.reset()

        return [obs]

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._rewards.append(reward)
        done = terminated or truncated
        if done:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None
        return [obs], reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


if __name__ == "__main__":
    env = AdroitHandHammer(render_mode="human")
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
