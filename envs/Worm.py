from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel


class Worm:
    ''' Unity Worm environment '''

    def __init__(self, file_name: str, worker_id: int, time_scale: int):
        '''
        Args:
            file_name {str} -- unity environment path
            worker_id {int} -- unity work id
            time_scale {int} -- unnity time scale
        '''
        self.file_name = file_name
        self.worker_id = worker_id
        self.channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
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


if __name__ == '__main__':
    import cv2
    file_name = 'UnityEnvs/Worm'
    env = Worm(file_name=file_name, worker_id=10101, time_scale=1)
    while True:
        done = False
        obs = env.reset()
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            # cv2.imshow('Video', cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
            # cv2.waitKey(1)
