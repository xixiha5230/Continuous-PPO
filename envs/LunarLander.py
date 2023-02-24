import gymnasium


class LunarLander:
    def __init__(self, action_type='discrete', render_mode=None):
        self._env = gymnasium.make('LunarLander-v2',
                                   continuous=True if action_type == 'continuous' else False,
                                   render_mode=render_mode)

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
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
