import argparse
import gym
from algorithm.Trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    # default="configs/LunarLander-v2.yaml",
    default="PPO_logs/LunarLander-v2/run_1/config.yaml",
    help="The config file",
)

# TODO
# 添加 tensorboard 完成
# 参数yaml固定化，保存运行时参数到新的yaml 完成
# 添加 ctrl-c监听 和 断点继续训练 完成
# 使用 dist 传递所有参数 完成
# 添加 lstm 
# 添加 图像、雷达
# 添加 多线程
# 添加 任务相关知识
if __name__ == '__main__':
    args = parser.parse_args()
    env = gym.make("LunarLander-v2", continuous=True)
    trainer = Trainer(env, args)
    trainer.run()
