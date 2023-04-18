import multiprocessing
import multiprocessing.connection
import sys

from utils.ConfigHelper import ConfigHelper
from utils.env_helper import create_env


def worker_process(remote: multiprocessing.connection.Connection, conf: ConfigHelper, id: int) -> None:
    '''Executes the threaded interface to the environment.

    Args:
        remote {multiprocessing.connection.Connection} -- Parent thread
        env_name {str} -- Name of the to be instantiated environment
        action_type {str} -- continuous or discrete. Action type of some environment
        id {int} -- worker id for unity environment
    '''
    # Spawn environment
    try:
        env = create_env(conf=conf, id=id)
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'close':
                remote.send(env.close())
                remote.close()
                break
            else:
                raise NotImplementedError(cmd)
        except KeyboardInterrupt:
            break
        except Exception as e:
            raise WorkerException(e)


class Worker:
    '''A worker that runs one environment on one processer.'''
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process

    def __init__(self, conf: ConfigHelper, id: int):
        '''
        Args:
            env_name (str) -- Name of the to be instantiated environment
            action_type (str) -- continuous or discrete. Action type of some environment
            id (int) -- worker id for unity environment
        '''
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, conf, id))
        self.process.start()


class WorkerException(Exception):
    '''Exception that is raised in the worker process and re-raised in the main process.'''

    def __init__(self, ee):
        self.ee = ee
        __,  __, self.tb = sys.exc_info()
        super(WorkerException, self).__init__(str(ee))

    def re_raise(self):
        raise (self.ee, None, self.tb)
