import os
import sys

import torch
import yaml


class ConfigHelper:
    def __init__(self, config_file) -> None:
        # load config
        with open(config_file, "r") as infile:
            self.conf: dict = yaml.safe_load(infile)

        # check device
        print(
            "============================================================================================"
        )
        if torch.cuda.is_available():
            device = "cuda"
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            device = "cpu"
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )

        # initialize train hyperparameters
        conf_train = {}
        conf_train: dict = self.conf.setdefault("train", conf_train)
        self.exp_name = conf_train.setdefault("exp_name", "default")
        self.K_epochs = conf_train.setdefault("K_epochs", 80)
        self.device = conf_train.setdefault("device", device)
        self.save_model_freq = conf_train.setdefault("save_model_freq", 5)
        self.random_seed = conf_train.setdefault("random_seed", 0)
        self.use_reward_scaling = conf_train.setdefault("use_reward_scaling", True)
        self.max_updates = conf_train.setdefault("max_updates", 150)
        self.num_mini_batch = conf_train.setdefault("num_mini_batch", 4)
        self.hidden_layer_size = conf_train.setdefault("hidden_layer_size", 256)
        self.update = conf_train.setdefault("update", 0)
        self.i_episode = conf_train.setdefault("i_episode", 0)
        self.resume = conf_train.setdefault("resume", False)
        self.run_num = conf_train.setdefault("run_num", 0)
        self.multi_task = conf_train.setdefault("multi_task", False)
        self.use_rnd = conf_train.setdefault("use_rnd", False)
        self.rnd_rate = conf_train.setdefault("rnd_rate", 0.5)
        self.use_state_normailzation = conf_train.setdefault(
            "use_state_normailzation", True
        )
        # PPO hyperparameters
        conf_ppo = {}
        conf_ppo: dict = self.conf.setdefault("ppo", conf_ppo)
        self.gamma = conf_ppo.setdefault("gamma", 0.99)
        self.lamda = conf_ppo.setdefault("lamda", 0.95)
        self.vf_loss_coeff = conf_ppo.setdefault("vf_loss_coeff", 0.5)
        self.entropy_coeff_schedule = {}
        self.entropy_coeff_schedule: dict = conf_ppo.setdefault(
            "entropy_coeff_schedule", self.entropy_coeff_schedule
        )
        self.entropy_coeff_schedule.setdefault("init", 0.001)
        self.entropy_coeff_schedule.setdefault("final", 0.001)
        self.entropy_coeff_schedule.setdefault("pow", 1.0)
        self.entropy_coeff_schedule.setdefault("max_decay_steps", 0)
        self.lr_schedule = {}
        self.lr_schedule: dict = conf_ppo.setdefault("lr_schedule", self.lr_schedule)
        self.lr_schedule.setdefault("init", 3.0e-4)
        self.lr_schedule.setdefault("final", 3.0e-4)
        self.lr_schedule.setdefault("pow", 1.0)
        self.lr_schedule.setdefault("max_decay_steps", 0)
        self.clip_range_schedule = {}
        self.clip_range_schedule: dict = conf_ppo.setdefault(
            "clip_range_schedule", self.clip_range_schedule
        )
        self.clip_range_schedule.setdefault("init", 0.2)
        self.clip_range_schedule.setdefault("final", 0.2)
        self.clip_range_schedule.setdefault("pow", 1.0)
        self.clip_range_schedule.setdefault("max_decay_steps", 0)
        self.task_schedule = {}
        self.task_schedule: dict = conf_ppo.setdefault(
            "task_schedule", self.task_schedule
        )
        self.task_schedule.setdefault("init", 1)
        self.task_schedule.setdefault("final", 0)
        self.task_schedule.setdefault("pow", 1.0)
        self.task_schedule.setdefault("max_decay_steps", 100)

        # LSTM hyperparameters
        recurrence = {}
        conf_recurrence: dict = self.conf.setdefault("recurrence", recurrence)
        self.use_lstm = conf_recurrence.setdefault("use_lstm", False)
        self.sequence_length = conf_recurrence.setdefault("sequence_length", -1)
        self.hidden_state_size = conf_recurrence.setdefault("hidden_state_size", 64)
        self.layer_type = conf_recurrence.setdefault("layer_type", "gru")
        self.reset_hidden_state = conf_recurrence.setdefault("reset_hidden_state", True)

        # Worker hyperparameters
        worker = {}
        conf_worker: dict = self.conf.setdefault("worker", worker)
        self.num_workers = conf_worker.setdefault("num_workers", 6)
        self.worker_steps = conf_worker.setdefault("worker_steps", 1000)

        # Enviroument settings
        env = {}
        conf_env: dict = self.conf.setdefault("env", env)
        self.env_name = conf_env["env_name"]
        self.env_type = conf_env.setdefault("env_type", "gym")
        self.env_win_path = conf_env.setdefault("env_win_path", "")
        self.env_linux_path = conf_env.setdefault("env_linux_path", "")
        self.env_action_type = conf_env.setdefault("env_action_type", "continuous")
        self.task_num = (
            max(len(self.env_linux_path), len(self.env_win_path))
            if self.multi_task
            else 1
        )
        self.worker_per_task = self.num_workers // self.task_num

        # drq hyperparameters
        drq = {}
        conf_drq: dict = self.conf.setdefault("drq", drq)
        self.use_drq = conf_drq.setdefault("use_drq", False)
        self.drq_image_pad = conf_drq.setdefault("drq_image_pad", 4)
        self.drq_ray_pad = conf_drq.setdefault("drq_ray_pad", 4)
        self.drq_M = conf_drq.setdefault("drq_M", 2)

        # config dir is : log_dir/env_name/exp_name/run_num/config.yaml,so global_dir is : dir of "log_dir"
        config_file_path = os.path.abspath(config_file)
        path_parts = config_file_path.split(os.sep)
        # -5 out of range, it's return "", means current dir
        self.glob_dir = os.sep.join(path_parts[:-5])

    def save(self, log_dir):
        self.conf["train"]["update"] = self.update
        self.conf["train"]["i_episode"] = self.i_episode
        self.conf["train"]["resume"] = True
        self.conf["train"]["run_num"] = self.run_num
        yaml_file = os.path.join(log_dir, "config.yaml")
        print(f"save configures at: {yaml_file}")
        with open(yaml_file, "w") as fp:
            yaml.dump(self.conf, fp)
