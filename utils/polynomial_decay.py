from utils.ConfigHelper import ConfigHelper


def get_decay(conf: ConfigHelper):
    learning_rate = polynomial_decay(conf.lr_schedule['init'], conf.lr_schedule['final'],
                                     conf.lr_schedule['max_decay_steps'], conf.lr_schedule['pow'], conf.update)
    clip_range = polynomial_decay(conf.clip_range_schedule['init'], conf.clip_range_schedule['final'],
                                  conf.clip_range_schedule['max_decay_steps'], conf.clip_range_schedule['pow'], conf.update)
    entropy_coeff = polynomial_decay(conf.entropy_coeff_schedule['init'], conf.entropy_coeff_schedule['final'],
                                     conf.entropy_coeff_schedule['max_decay_steps'], conf.entropy_coeff_schedule['pow'], conf.update)
    task_coeff = polynomial_decay(conf.task_schedule['init'], conf.task_schedule['final'],
                                  conf.task_schedule['max_decay_steps'], conf.task_schedule['pow'], conf.update)
    return learning_rate, clip_range, entropy_coeff, task_coeff


def polynomial_decay(initial: float, final: float, max_decay_steps: int, pow: float, current_step: int) -> float:
    """Decays hyperparameters polynomially. If pow is set to 1.0, the decay behaves linearly. 
    Args:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        pow {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training
    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return ((initial - final) * ((1 - current_step / max_decay_steps) ** pow) + final)
