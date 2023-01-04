import argparse
import gym
from algorithm.Trainer import Trainer
# ref
# https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt.git
# https://github.com/nikhilbarhate99/PPO-PyTorch.git
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/BipedalWalker-v3.yaml",
    # default="PPO_logs/LunarLander-v2/init/run_3/config.yaml",
    help="The config file",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="test",
    help="The experiment name",
)

# TODO
# 添加 tensorboard 完成
# 添加 参数yaml固定化，保存运行时参数到新的yaml 完成
# 添加 ctrl-c监听 和 断点继续训练 完成
# 使用 dist 传递所有参数 完成
# 添加 多线程  完成
# 添加 lstm   完成

# 添加 雷达 完成 图像 进行中
# 添加 任务相关知识
if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.run()
    trainer.close()
