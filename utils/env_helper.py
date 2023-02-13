import gym
from envs.UnityWrapper import UnityWrapper
from envs.LunarLander import LunarLander
from envs.CartPole import CartPole
from envs.CarRacing import CarRacing
from envs.Walker2d import Walker2d


def create_env(env_name: str, continuous: bool = False, id: int = 0, render_mode=None):
    """Initializes an environment based on the provided environment name.

    Args:
        env_name {str}: Name of the to be instantiated environment
    Returns:
        {env}: Returns the selected environment instance.
    """
    if env_name == "LunarLander-v2":
        return LunarLander(continuous=continuous, render_mode=render_mode)
    elif env_name == 'BipedalWalker-v3':
        raise NotImplementedError()
        return gym.make("BipedalWalker-v3")
    elif env_name == 'CartPole-v1' or env_name == 'CartPole-v0':
        return CartPole(env_name, render_mode=render_mode)
    elif env_name == 'CarRacing-v1':
        return CarRacing(continuous=continuous, render_mode=render_mode)
    elif env_name == 'Walker2d-v4':
        return Walker2d(render_mode=render_mode)
    elif env_name == 'CarRace':
        return UnityWrapper(file_name="./UnityEnvs/CarRace", worker_id=id)
    else:
        raise f"Unknow env: {env_name}"
