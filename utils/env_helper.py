from utils.ConfigHelper import ConfigHelper


def create_env(
    conf: ConfigHelper, id: int = 0, render_mode=None, time_scale=100, seed=0
):
    """Initializes an environment based on the provided environment name.

    Args:
        conf {dict} -- configure file
        id (int) -- worker id for unity environment
        render_mode (str) --  human or rgb_arraty. Gym render mode.
        time_scale (int) -- time scale for unity environment, higher time scale can help speed up
    Returns:
        {env}: Returns the selected environment instance.
    """
    env_name = conf.env_name
    action_type = conf.action_type
    task = conf.task
    if env_name == None:
        from envs.UnityCommon import UnityCommon

        return UnityCommon(None, 0, 1)
    elif (
        env_name == "CarRace"
        or env_name == "CarRace_NoReset"
        or env_name == "CarSearch"
        or env_name == "CarSearch_NoReset"
        or env_name == "CarSearchCkpt"
        or env_name == "Pyramids"
        or env_name == "Hallway"
    ):
        from envs.UnityCommon import UnityCommon

        return UnityCommon(
            file_name=f"./UnityEnvs/{env_name}", worker_id=id, time_scale=time_scale
        )
    elif env_name == "UnityMultitask":
        from envs.UnityMultitask import UnityMultitask

        return UnityMultitask(task, id, time_scale)
    elif env_name == "LunarLander-v2":
        from envs.LunarLander import LunarLander

        return LunarLander(action_type=action_type, render_mode=render_mode)
    elif env_name == "BipedalWalker-v3":
        raise NotImplementedError()
    elif env_name == "MountainCar-v0" or env_name == "MountainCarContinuous-v0":
        from envs.MountainCar import MountainCar

        return MountainCar(env_name, render_mode=render_mode)
    elif env_name == "CartPole-v1" or env_name == "CartPole-v0":
        from envs.CartPole import CartPole

        return CartPole(env_name, render_mode=render_mode)

    elif env_name == "CarRacing-v1":
        from envs.CarRacing import CarRacing

        return CarRacing(action_type=action_type, render_mode=render_mode)
    elif env_name == "Walker2d-v4" or env_name == "Walker2d-v2":
        from envs.Walker2d import Walker2d

        return Walker2d(env_name, render_mode=render_mode)
    elif env_name == "HopperJump":
        from envs.Hoper.HopperJump import HopperJump

        return HopperJump("mo-hopper-v4", render_mode=render_mode)
    elif env_name == "HopperRun":
        from envs.Hoper.HopperRun import HopperRun

        return HopperRun("mo-hopper-v4", render_mode=render_mode)
    elif env_name == "HalfCheetahVel":
        from envs.HalfCheetah.HalfCheetahVel import HalfCheetahVel

        return HalfCheetahVel(task=task, render_mode=render_mode, id=id)
    elif env_name == "HalfCheetahDir":
        from envs.HalfCheetah.HalfCheetahDir import HalfCheetahDir

        return HalfCheetahDir(task=task, render_mode=render_mode, id=id)
    elif "MiniGrid" in env_name:
        from envs.MinigridMemory import Minigrid

        return Minigrid(env_name, render_mode)
    elif env_name == "AdroitHandHammer-v1":
        from envs.AdroitHandHammer import AdroitHandHammer

        return AdroitHandHammer(render_mode=render_mode)
    else:
        raise f"Unknow env: {env_name}"
