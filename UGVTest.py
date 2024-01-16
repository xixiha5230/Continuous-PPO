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
        state = self.obs_preprocess(obs)

        while True:
            action = self.select_action(state)
            obs, reward, done, _ = self.env.step(action)
            state = self.obs_preprocess(obs)
            self.env.render()
            time.sleep(0.05)
            if done:
                obs = self.env.reset()
                self.reset()
                state = self.obs_preprocess(obs)


if __name__ == "__main__":
    UGVTest("/mnt/d/Code/PPO/PPO_logs/UGVRace/new_env/run_1/config.yaml")
