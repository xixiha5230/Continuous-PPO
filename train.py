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
    default="configs/LunarLander-v2.yaml",
    # default="PPO_logs/LunarLander-v2/lstm/run_6/config.yaml",
    help="The config file",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default="test",
    help="The experiment name",
)

if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.run()
    trainer.close()
