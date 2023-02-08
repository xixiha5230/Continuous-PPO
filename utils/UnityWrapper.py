from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class UnityWrapper:
    def __init__(self, file_name, worker_id):
        self.file_name = file_name
        self.worker_id = worker_id
        self.channel = EngineConfigurationChannel()
        unity_env = UnityEnvironment(file_name=file_name,
                                     worker_id=worker_id,
                                     side_channels=[self.channel])
        self.env = UnityToGymWrapper(unity_env,
                                     uint8_visual=False,
                                     flatten_branched=True,
                                     allow_multiple_obs=True)
        self.channel.set_configuration_parameters(
            width=256,
            height=256,
            quality_level=5,
            time_scale=5,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    def seed(self, seed):
        return self.env.seed(seed)
