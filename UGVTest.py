import time

from TestMain import TestMain
from utils.ConfigHelper import ConfigHelper
from utils.env_helper import create_env


class UGVTest(TestMain):
    def __init__(self, config_file: str, ckpt: int = None):
        conf = ConfigHelper(config_file)
        self.env = create_env(conf, time_scale=1)
        super().__init__(self.env.observation_space, self.env.action_space, conf, ckpt)

    def _run(self):
        obs = self.env.reset()
        self.reset()
        reward = 0
        while True:
            action = self.select_action(obs)
            obs, _r, done, _ = self.env.step(action)
            reward += _r
            self.env.render()
            time.sleep(0.01)
            if done:
                obs = self.env.reset()
                self.reset()
                print(f"reward: {reward}")
                reward = 0


if __name__ == "__main__":
    UGVTest("/mnt/d/Code/PPO/PPO_logs/UGVRace/new_env/run_10/config.yaml")
