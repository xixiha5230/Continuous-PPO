from enum import Enum


class WorkerCommand(Enum):
    """WorkerCommand Enum"""

    step = 1
    reset = 2
    close = 3
