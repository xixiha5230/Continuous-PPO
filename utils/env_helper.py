import os
import sys
from importlib import import_module

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
    env_name: str = conf.env_name
    env_type = conf.env_type
    env_win_path = conf.env_win_path
    env_linux_path = conf.env_linux_path
    action_type = conf.env_action_type
    if env_name == None:
        from envs.UnityCommon import UnityCommon

        return UnityCommon(None, 0, 1)
    elif env_type == "Unity":
        from envs.UnityCommon import UnityCommon

        env_path = env_win_path if sys.platform == "win32" else env_linux_path
        env_path = os.path.join(conf.glob_dir, env_path)
        return UnityCommon(file_name=env_path, worker_id=id, time_scale=time_scale)
    elif env_type == "Unity_UGV":
        from envs.UnityUGV import UnityUGV

        env_path = env_win_path if sys.platform == "win32" else env_linux_path
        return UnityUGV(file_name=env_path, worker_id=id, time_scale=time_scale)
    elif env_type == "Unity_Multitask":
        from envs.UnityMultitask import UnityMultitask

        return UnityMultitask(conf, id, time_scale)
    elif env_type == "gym" or env_type == "mujoco":
        class_name = env_name.split("-")[0]
        module = import_module(f"envs.{class_name}")
        gym_class = getattr(module, class_name)
        return gym_class(
            env_name=env_name, action_type=action_type, render_mode=render_mode, id=id
        )
    elif env_name == "HalfCheetahVel":
        from envs.HalfCheetah.HalfCheetahVel import HalfCheetahVel

        return HalfCheetahVel(task=None, render_mode=render_mode, id=id)
    else:
        raise f"Unknow env: {env_name}"
