import time
from typing import List

import numpy as np

from joystick import Joystick
from rosugv.ros_ugv import RL_CAMERA, RL_DEPTH, RL_LASER, RL_STATE, RosUGV
from TestMain import TestMain as Main
from utils.ConfigHelper import ConfigHelper

DEBUG_ACTION = False


class RosMain(Main):
    def __init__(
        self,
        config_dir,
        ckpt=None,
        rl_keys: List[str] = [RL_CAMERA, RL_LASER, RL_STATE],
        decision_period: int = 5,
        decision_rate: int = 20,
        motor_limit: float = 0.25,
        ema: float = 0.2,
    ):
        self.decision_period = decision_period
        self.decision_rate = decision_rate
        self.motor_limit = motor_limit
        self.ema = ema

        self.js = js = Joystick()
        js.run()

        self.conf = ConfigHelper(config_dir)
        self.ros_ugv = RosUGV(
            motor_limit=self.motor_limit,
            ema=self.ema,
            repeat_action=True,
            active_rl_keys_list=rl_keys,
        )
        obs_space = self._custom_rl_obs_space(*self.ros_ugv.get_rl_obs_shapes())
        action_space = self._custom_rl_action_space(self.ros_ugv.get_rl_action_size())

        super().__init__(obs_space, action_space, self.conf, ckpt)

    def _custom_rl_obs_space(self, *obs_shapes):
        return list(obs_shapes)

    def _custom_rl_action_space(self, action_shape):
        return action_shape

    def _custom_rl_obs_list(self, *obs_list):
        return list(obs_list)

    def _run(self):
        try:
            while True:
                self.ros_ugv.stop()
                print("Waiting...")

                while (
                    False
                    and not self.js.button_states["x"]
                    and not self.js.button_states["a"]
                    and not self.js.button_states["tl"]
                ):
                    time.sleep(0.1)

                if self.js.button_states["tl"]:
                    print("Closing...")
                    break

                if self.js.button_states["x"] or True:
                    print("Start Episode")
                    self.ros_ugv.reset()
                    self.reset()
                    obs_list = self.ros_ugv.get_rl_obs_list()
                    obs_list = self._custom_rl_obs_list(*obs_list)
                    obs_list = self.obs_preprocess(obs_list)
                    step = 0
                    while not self.js.button_states["y"]:
                        if step % self.decision_period == 0:
                            action = self.select_action(obs_list, is_ros=True)
                            if DEBUG_ACTION:
                                action = np.zeros((1, 2), dtype=np.float32)
                            print(action)
                            # [[转向,油门]]
                            action = [[action[0][0], 0.5 * action[0][1]]]

                        self.ros_ugv.send_rl_action(action)
                        step += 1
                        obs_list = self.ros_ugv.get_rl_obs_list()
                        obs_list = self._custom_rl_obs_list(*obs_list)
                        obs_list = self.obs_preprocess(obs_list)
                    self.ros_ugv.stop()

                elif self.js.button_states["a"]:
                    print("Start Manual")

                    step = 0

                    t = time.time()

                    while not self.js.button_states["y"]:
                        if step % self.decision_period == 0:
                            action = np.array(
                                [self.js.axis_states["rx"], -self.js.axis_states["y"]]
                            )
                            action = np.expand_dims(action, 0)

                        self.ros_ugv.send_rl_action(action)

                        while time.time() - t < 1 / self.decision_rate:
                            time.sleep(0.001)

                        step += 1
                        t = time.time()

                    self.ros_ugv.stop()

        finally:
            self.js.close()
            self.ros_ugv.close()
