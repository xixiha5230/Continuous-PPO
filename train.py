import argparse

from trainer.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    type=str,
    default="configs/UnityMultitask.yaml",
    # default="PPO_logs/UnityMultitask/gru_seq16/run_0/config.yaml",
    help='The config file',
)

# TODO
# 任务表征网络的必要性
# 优化整理代码
if __name__ == '__main__':
    args = parser.parse_args()
    for _ in range(3):
        trainer = Trainer(args.config_file)
        trainer.run()
        trainer.close()
        if trainer.conf.resume:
            break
