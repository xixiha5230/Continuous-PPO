import argparse
from trainer.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    type=str,
    default='configs/CarSearch.yaml',
    help='The config file',
)
parser.add_argument(
    '--exp_name',
    type=str,
    default='test',
    help='The experiment name',
)

# TODO
# 优化整理代码
if __name__ == '__main__':
    args = parser.parse_args()
    for _ in range(10):
        trainer = Trainer(args.config_file, args.exp_name)
        trainer.run()
        if trainer.stop_signal:
            break
        else:
            trainer.close()
