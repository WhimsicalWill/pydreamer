import torch
import torch.nn as nn

from .a2c import *
from .networks import *

class ImagBehavior(nn.Module):

    def __init__(self, conf, state_dim, world_model):
        
        self.wm = world_model

        # Actor critic

        self.ac = ActorCritic(in_dim=state_dim,
                              out_actions=conf.action_dim,
                              layer_norm=conf.layer_norm,
                              gamma=conf.gamma,
                              lambda_gae=conf.lambda_gae,
                              entropy_weight=conf.entropy,
                              target_interval=conf.target_interval,
                              actor_grad=conf.actor_grad,
                              actor_dist=conf.actor_dist,
                              )

    def training_step(self, in_state_dream, obs, iwae_samples, imag_horizon, metrics, tensors, reward_func, goal_embed=None):
        T, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon

        # Note features_dream includes the starting "real" features at features_dream[0]
        features_dream, actions_dream, rewards_dream, terminals_dream = \
            self.dream(in_state_dream, H, self.ac.actor_grad == 'dynamics', goal_embed)  # (H+1,TBI,D)
        (loss_actor, loss_critic), metrics_ac, tensors_ac = \
            self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream)
        metrics.update(**metrics_ac)
        tensors.update(policy_value=unflatten_batch(tensors_ac['value'][0], (T, B, I)).mean(-1))
        return loss_actor, loss_critic

    # unrolls a policy in imagination using the world model
    def dream(self, in_state: StateB, imag_horizon: int, reward_func, dynamics_gradients=False, goal_embed=None):
        features = []
        actions = []
        state = in_state
        self.wm.requires_grad_(False)  # Prevent dynamics gradiens from affecting world model

        for i in range(imag_horizon):
            feature = self.wm.core.to_feature(*state)
            if goal_embed:
                action_dist = self.ac.forward_actor(feature, goal_embed)
            else:
                action_dist = self.ac.forward_actor(feature)
            if dynamics_gradients:
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
            features.append(feature)
            actions.append(action)
            # When using dynamics gradients, this causes gradients in RSSM, which we don't want.
            # This is handled in backprop - the optimizer_model will ignore gradients from loss_actor.
            _, state = self.wm.core.cell.forward_prior(action, None, state)

        feature = self.wm.core.to_feature(*state)
        features.append(feature)
        features = torch.stack(features)  # (H+1,TBI,D)
        actions = torch.stack(actions)  # (H,TBI,A)

        # TODO: add different reward computation based on if there is goal-conditioning
        # maybe we could pass in the function that computes rewards as an argument
        # for both achiever and explorer, we don't train a reward decoder

        # rewards = self.wm.decoder.reward.forward(features)      # (H+1,TBI)
        rewards = reward_func(rewards)                          # (H+1,TBI
        terminals = self.wm.decoder.terminal.forward(features)  # (H+1,TBI)

        self.wm.requires_grad_(True)
        return features, actions, rewards, terminals

class GCDreamerBehavior(ImagBehavior):
    '''
    Goal conditioned Dreamer behavior
    '''

    def __init__(self, conf, state_dim, world_model):
        super(GCDreamerBehavior, self).__init__(conf, state_dim, world_model)

    def forward(self, feature, goal_embed):
        action_distr = self.ac.forward_actor(feature, goal_embed)  # (1,B,A)
        value = self.ac.forward_value(feature, goal_embed)  # (1,B)
        return action_distr, value


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

    def forward(self, feature):
        action_distr = self.ac.forward_actor(feature)  # (1,B,A)
        value = self.ac.forward_value(feature)  # (1,B)
        return action_distr, value

    # this functions extends the ImagBehavior training_step() function, and adds extra training steps
    def training_step(self, in_state_dream, obs, iwae_samples, imag_horizon, metrics, tensors):
        T, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon

        # TODO: implement training for ensemble (must feed in targets similar to world model training)
        # TODO: implement training for behavior
        # TODO: eventually aggregate metrics for tracking
        # self._train_ensemble(feat, target)

        # The following line is replaced by the dream() and training_step() functions below
        # self._behavior.train(start, self._intrinsic_reward)[-1]

        # Note features_dream includes the starting "real" features at features_dream[0]
        features_dream, actions_dream, rewards_dream, terminals_dream = \
            self.dream(in_state_dream, H, self.ac.actor_grad == 'dynamics')  # (H+1,TBI,D)
        (loss_actor, loss_critic), metrics_ac, tensors_ac = \
            self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream)
        return loss_actor, loss_critic

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