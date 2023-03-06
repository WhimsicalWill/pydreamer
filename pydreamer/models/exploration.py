import torch
import torch.nn as nn
import torch.nn.functional as F

from .behaviors import *
from .networks import *

class Plan2Explore(nn.Module):
    '''
    Defines an ensemble of models used for exploration.
    '''
    def __init__(self,
                 conf, 
                 world_model, 
                 reward=None
                 ):
        self._conf = conf
        self._reward = reward
        self._behavior = ImagBehavior(conf, world_model)
        self.actor = self._behavior.actor
        size = {
            'embed': 32 * conf.cnn_depth,
            'stoch': conf.dyn_stoch,
            'deter': conf.dyn_deter,
            'feat': conf.dyn_stoch + conf.dyn_deter,
        }[self._conf.disag_target]
        kw = dict(
            shape=size, layers=conf.disag_layers, units=conf.disag_units,
            act=conf.act
        )
        self._networks = nn.ModuleList([DenseHead(**kw) for _ in range(conf.disag_models)])
        self.optimizer = torch.optim.Adam(
            self._networks.parameters(), lr=conf.model_lr, 
            weight_decay=conf.weight_decay
        )

    def train(self, start, feat, embed, kl):
        metrics = {}
        target = {
            'embed': embed,
            'stoch': start['stoch'],
            'deter': start['deter'],
            'feat': feat,
        }[self._conf.disag_target]
        metrics.update(self._train_ensemble(feat, target))
        metrics.update(self._behavior.train(start, self._intrinsic_reward)[-1])
        return None, metrics

    def _intrinsic_reward(self, feat, state, action):
        pred = torch.stack([head(feat) for head in self._networks], dim=0)
        variance = torch.var(pred, dim=0)
        disagreement = torch.mean(variance)
        reward = self._conf.expl_intr_scale * disagreement
        if self._conf.expl_extr_scale:
            reward += self._conf.expl_extr_scale * \
                      self._reward(feat, state, action)
        return reward

    def _train_ensemble(self, inputs, targets):
        if self._conf.disag_offset:
            targets = targets[:, self._conf.disag_offset:]
            inputs = inputs[:, :-self._conf.disag_offset]
        targets.detach()
        inputs.detach()
        preds = torch.stack([head(inputs) for head in self._networks], dim=0)
        likes = [torch.mean(preds.log_prob(targets), dim=-1)]
        loss = -torch.sum(likes)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'model_loss': loss.item()}

    def act(self, feat, *args):
        return self._actor(feat)
