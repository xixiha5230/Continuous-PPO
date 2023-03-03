from envs.CarRace import CarRace
from envs.LunarLander import LunarLander
from envs.CartPole import CartPole
from envs.CarRacing import CarRacing
from envs.CarSearch import CarSearch
from envs.Walker2d import Walker2d
from envs.MountainCar import MountainCar
from envs.Hallway import Hallway
from envs.MinigridMemory import Minigrid
from envs.Hoper.HopperJump import HopperJump
from envs.Hoper.HopperRun import HopperRun


def create_env(env_name: str, action_type: bool = False, id: int = 0, render_mode=None, time_scale=2):
    '''Initializes an environment based on the provided environment name.

    Args:
        env_name {str}: Name of the to be instantiated environment
        action_type (str) -- continuous or discrete. Action type of some environment
        id (int) -- worker id for unity environment
        render_mode (str) --  human or rgb_arraty. Gym render mode.
        time_scale (int) -- time scale for unity environment, higher time scale can help speed up
    Returns:
        {env}: Returns the selected environment instance.
    '''
    if env_name == 'LunarLander-v2':
        return LunarLander(action_type=action_type, render_mode=render_mode)
    elif env_name == 'BipedalWalker-v3':
        raise NotImplementedError()
    elif env_name == 'MountainCar-v0' or env_name == 'MountainCarContinuous-v0':
        return MountainCar(env_name)
    elif env_name == 'CartPole-v1' or env_name == 'CartPole-v0':
        return CartPole(env_name, render_mode=render_mode)
    elif env_name == 'CarRacing-v1':
        return CarRacing(action_type=action_type, render_mode=render_mode)
    elif env_name == 'Walker2d-v4' or env_name == 'Walker2d-v2':
        return Walker2d(env_name, render_mode=render_mode)
    elif env_name == 'CarRace' or env_name == 'CarRace_NoReset':
        return CarRace(file_name=f'./UnityEnvs/{env_name}', worker_id=id, time_scale=time_scale)
    elif env_name == 'CarSearch' or env_name == 'CarSearch_NoReset' or env_name == 'CarSearchCkpt':
        return CarSearch(file_name=f'./UnityEnvs/{env_name}', worker_id=id, time_scale=time_scale)
    elif env_name == 'Hallway':
        return Hallway(file_name=f'./UnityEnvs/{env_name}', worker_id=id, time_scale=time_scale, render_mode=render_mode)
    elif env_name == 'HopperJump':
        return HopperJump('mo-hopper-v4', render_mode=render_mode)
    elif env_name == 'HopperRun':
        return HopperRun('mo-hopper-v4', render_mode=render_mode)
    elif "MiniGrid" in env_name:
        return Minigrid(env_name, render_mode)
    else:
        raise f'Unknow env: {env_name}'
