from gym.spaces import Box, Tuple

from envs.UnityCommon import UnityCommon


class UnityUGV(UnityCommon):
    """Unity car race environment"""

    def __init__(self, file_name: str, worker_id: int, time_scale: int, seed: int = 0):
        super().__init__(file_name, worker_id, time_scale, seed)

    @property
    def observation_space(self):
        obs_space = Tuple(
            (
                self._env.observation_space[0],
                Box(
                    float("-inf"),
                    float("+inf"),
                    (self._env.observation_space[1].shape[0] - 2,),
                ),
                Box(
                    float("-inf"),
                    float("+inf"),
                    (2,),
                ),
            )
        )
        return obs_space

    def reset(self):
        self._rewards = []
        obs = self._env.reset()
        obs = [obs[0], obs[1][:-2], obs[1][-2:]]
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        if done:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}
        else:
            info = None
        obs = [obs[0], obs[1][:-2], obs[1][-2:]]
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


if __name__ == "__main__":
    import cv2

    file_name = "UnityEnvs/maUGVRace"
    env = UnityCommon(file_name=file_name, worker_id=0, time_scale=1)
    while True:
        done = False
        obs = env.reset()
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            cv2.imshow("Video", cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
