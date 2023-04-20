import gymnasium


class MountainCar:
    ''' Gym MountainCar environment '''

    def __init__(self, env_name, render_mode=None):
        '''
        Args:
            name {str} -- name of diffrent version
            render_mode {str} -- render mode: humnan or rgb_array
        '''
        self._env = gymnasium.make(env_name, render_mode=render_mode)

    @property
    def observation_space(self):
        return self._env.observation_space

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
            info = {'reward': sum(self._rewards),
                    'length': len(self._rewards)}
        else:
            info = None
        return [obs], reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
