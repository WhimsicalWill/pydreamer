import torch
import torch.nn as nn
import torch.nn.functional as F
import networks

from .behaviors import *

class Plan2Explore(nn.Module):
    '''
    Defines an ensemble of models used for exploration.
    '''
    def __init__(self, config, world_model, reward=None):
        self._config = config
        self._reward = reward
        self._behavior = ImagBehavior(config, world_model)
        self.actor = self._behavior.actor
        size = {
            'embed': 32 * config.cnn_depth,
            'stoch': config.dyn_stoch,
            'deter': config.dyn_deter,
            'feat': config.dyn_stoch + config.dyn_deter,
        }[self._config.disag_target]
        kw = dict(
            shape=size, layers=config.disag_layers, units=config.disag_units,
            act=config.act
        )
        self._networks = nn.ModuleList([networks.DenseHead(**kw) for _ in range(config.disag_models)])
        self.optimizer = torch.optim.Adam(
            self._networks.parameters(), lr=config.model_lr, 
            weight_decay=config.weight_decay
        )

    def train(self, start, feat, embed, kl):
        metrics = {}
        target = {
            'embed': embed,
            'stoch': start['stoch'],
            'deter': start['deter'],
            'feat': feat,
        }[self._config.disag_target]
        metrics.update(self._train_ensemble(feat, target))
        metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        pred = torch.stack([head(feat) for head in self._networks], dim=0)
        variance = torch.var(pred, dim=0) # TODO: should we use biased or unbiased variance?
        disagreement = torch.mean(variance)
        reward = self._config.expl_intr_scale * disagreement
        if self._config.expl_extr_scale:
            reward += self._config.expl_extr_scale * \
                      self._reward(feat, state, action)
        return reward

    def _train_ensemble(self, inputs, targets):
        if self._config.disag_offset:
            targets = targets[:, self._config.disag_offset:]
            inputs = inputs[:, :-self._config.disag_offset]
        targets.detach()
        inputs.detach()
        preds = torch.stack([head(inputs) for head in self._networks], dim=0)
        likes = [torch.mean(preds.log_prob(targets), dim=-1)]
        loss = -torch.sum(likes)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'disagreement': loss.item()} # TODO: what should this return?

    def act(self, feat, *args):
        return self._actor(feat)
