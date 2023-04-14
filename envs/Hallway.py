from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel


class Hallway:
    ''' Unity Hallway environment '''

    def __init__(self, file_name, worker_id, time_scale, render_mode=None):
        '''
        Args:
            file_name {str} -- unity environment path
            worker_id {int} -- unity work id
            time_scale {int} -- unnity time scale
            render_mode {str} -- render mode: only set humnan to render
        '''
        self.file_name = file_name
        self.worker_id = worker_id
        self.channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
                                     no_graphics=False if render_mode == 'human' else True,
                                     side_channels=[self.channel])
        self._env = UnityToGymWrapper(unity_env,
                                      uint8_visual=False,
                                      flatten_branched=True,
                                      allow_multiple_obs=True)
        self.channel.set_configuration_parameters(
            width=512,
            height=384,
            quality_level=5,
            time_scale=time_scale,
        )

    @property
    def observation_space(self):
        return self._env.observation_space[0]

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        return obs[0]

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        if done:
            info = {'reward': sum(self._rewards),
                    'length': len(self._rewards)}
        else:
            info = None
        return obs[0], reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()
