import glob
import os

import torch

from algorithm.PPO import PPO
from utils.ConfigHelper import ConfigHelper
from utils.Logger import Logger
from utils.obs_2_tensor import _obs_2_tensor
from utils.recurrent_cell_init import recurrent_cell_init


class TestMain:
    def __init__(
        self,
        obs_space: tuple,
        action_space: tuple,
        conf: ConfigHelper,
        ckpt: int = None,
    ):
        self.conf = conf
        self.logger = Logger(self.conf, True)
        self.state_normalizer = self.logger.load_pickle("state_normalizer.pkl")
        self.state_normalizer.config = self.conf
        self.agent = PPO(obs_space, action_space, self.conf)
        latest_checkpoint = (
            max(
                glob.glob(os.path.join(self.logger.checkpoint_path, "*")),
                key=os.path.getctime,
            )
            if ckpt is None
            else os.path.join(self.logger.checkpoint_path, f"{ckpt}.pth")
        )

        print(f"resume from {latest_checkpoint}")
        self.agent.load(latest_checkpoint)
        self.agent.policy.eval()
        self.h_in = recurrent_cell_init(
            1, self.conf.hidden_state_size, self.conf.layer_type, self.conf.device
        )
        with torch.no_grad():
            self._run()

    def obs_preprocess(self, obs):
        state = self.state_normalizer(obs, update=False)
        state = _obs_2_tensor(state, self.conf.device)
        if len(state[-1].shape) < 2:
            state = [s.unsqueeze(0) for s in state]
        return state

    def reset(self):
        self.h_in = recurrent_cell_init(
            1, self.conf.hidden_state_size, self.conf.layer_type, self.conf.device
        )

    def select_action(self, state):
        action, self.h_in, _ = self.agent.eval_select_action(state, self.h_in)
        return action.cpu().numpy()

    def _run():
        raise NotImplementedError
