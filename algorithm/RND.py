import torch
import torch.nn as nn

from layers.StateNet import AtariImage
from utils.weights_init import weights_init_


class RND(nn.Module):
    def __init__(self, config: dict, obs_shape: tuple) -> None:
        super(RND, self).__init__()
        self.config = config
        conf_train = config['train']
        self.device = conf_train['device']
        conf_worker = config['worker']
        self.n_workers = conf_worker['num_workers']
        self.worker_steps = conf_worker['worker_steps']
        conf_ppo = config['ppo']
        self.gamma = conf_ppo['gamma']
        self.lamda = conf_ppo['lamda']
        # only RND image or vector obs
        if len(obs_shape) == 3:
            _dumy = AtariImage(obs_shape)
            self.target_net = nn.Sequential(
                AtariImage(obs_shape),
                nn.Linear(_dumy.output_size, 512)
            )
            self.target_net.apply(weights_init_)
            for p in self.target_net.parameters():
                p.requires_grad = False

            self.predict_net = nn.Sequential(
                AtariImage(obs_shape),
                nn.Linear(_dumy.output_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )
            self.predict_net.apply(weights_init_)

        elif len(obs_shape) == 1:
            raise NotImplementedError('vector RND NotImplemented !')
        else:
            raise NotImplementedError(obs_shape)

    def calculate_rnd_loss(self, next_state):
        encoded_target_features = self.forward_target(next_state)
        encoded_predictor_features = self.forward_predict(next_state)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)
        # TODO mask的作用
        predictor_proportion = 32. / self.n_workers
        mask = torch.rand(loss.size()).to(self.device)
        mask = (mask < predictor_proportion).float()
        loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        return loss

    def calculate_rnd_rewards(self, next_states: torch.Tensor, batch=True):
        # next_states shape like (num_worker,step, features)
        if not batch:
            next_states = next_states.unsqueeze(0)
        else:
            orig_shape = next_states.size()
            new_shape = (orig_shape[0] * orig_shape[1],) + orig_shape[2:]
            next_states = next_states.view(new_shape)

        predictor_encoded_features = self.forward_predict(next_states)
        target_encoded_features = self.forward_target(next_states)

        rnd_reward = (predictor_encoded_features - target_encoded_features).pow(2).mean(1)
        if not batch:
            return rnd_reward.detach().cpu().numpy()
        else:
            return rnd_reward.detach().cpu().numpy().reshape((orig_shape[0], orig_shape[1]))

    def forward_target(self, inputs: torch.Tensor):
        outputs = self.target_net(inputs)
        return outputs

    def forward_predict(self, inputs: torch.Tensor):
        outputs = self.predict_net(inputs)
        return outputs
