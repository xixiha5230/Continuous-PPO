import argparse
from trainer.Trainer import Trainer

# *******       ref       *******
# https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt.git    lstm + multiprocessor + buffer
# https://github.com/nikhilbarhate99/PPO-PyTorch.git                ppo
# https://github.com/seungeunrho/minimalRL                          lstm + ppo
# https://github.com/BlueFisher/Advanced-Soft-Actor-Critic          conv1d + conv2d
# https://github.com/Lizhi-sjtu/DRL-code-pytorch                    optimization

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='configs/Walker2d-v4.yaml',
    # default=f'PPO_logs/Walker2d-v4/lstm_continue/run_1/config.yaml',
    help='The config file',
)
parser.add_argument(
    '--exp_name',
    type=str,
    default='test',
    help='The experiment name',
)
# TODO
# add ICM
if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.run()
    trainer.close()
