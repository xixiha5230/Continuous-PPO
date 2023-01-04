import gym


def create_env(env_name: str, continuous: bool = False):
    """Initializes an environment based on the provided environment name.

    Args:
        env_name {str}: Name of the to be instantiated environment
    Returns:
        {env}: Returns the selected environment instance.
    """
    if env_name == "LunarLander-v2":
        return gym.make(env_name, continuous=continuous)
    elif env_name == 'BipedalWalker-v3':
        return gym.make("BipedalWalker-v3")
    else:
        raise f"Unknow env: {env_name}"
