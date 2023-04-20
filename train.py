import argparse

from trainer.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    type=str,
    default='configs/LunarLander-v2.yaml',
    help='The config file',
)

# TODO
# 任务表征网络的必要性
# 优化整理代码
if __name__ == '__main__':
    args = parser.parse_args()
    for _ in range(10):
        trainer = Trainer(args.config_file)
        trainer.run()
        trainer.close()
        if trainer.conf.resume:
            break
