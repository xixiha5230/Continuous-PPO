import argparse
from trainer.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    type=str,
    default='configs/CartPole-v1.yaml',
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
