import torch
import torch.nn as nn

from .a2c import *
from .networks import *

class ImagBehavior(nn.Module):

    def __init__(self, conf, world_model, use_gc=False):
        
        self.wm = world_model

        # Actor critic

        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.ac = ActorCritic(in_dim=state_dim,
                              goal_dim=self.wm.encoder.out_dim if use_gc else 0,
                              out_actions=conf.action_dim,
                              layer_norm=conf.layer_norm,
                              gamma=conf.gamma,
                              lambda_gae=conf.lambda_gae,
                              entropy_weight=conf.entropy,
                              target_interval=conf.target_interval,
                              actor_grad=conf.actor_grad,
                              actor_dist=conf.actor_dist,
                              )

        # TODO: we may want to add a separate config argument for actor_grad for explorer/achiever actors

    # TODO: maybe refactor this to train(), to be more intuitive and match LEXA better
    def training_step(self, in_state_dream, H, reward_func, goal_embed=None):
        # Note features_dream includes the starting "real" features at features_dream[0]
        features_dream, actions_dream, rewards_dream, terminals_dream = \
            self.dream(in_state_dream, H, self.conf.actor_grad == 'dynamics', reward_func, goal_embed)  # (H+1,TBI,D)

        (loss_actor, loss_critic), metrics_ac, tensors_ac = \
            self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream)
        return (loss_actor, loss_critic), features_dream, actions_dream, rewards_dream, terminals_dream, tensors_ac

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

        if goal_embed:
            rewards = reward_func(features, goal_embed) # (H+1,TBI)
        else:
            rewards = reward_func(features)             # (H+1,TBI)
        terminals = self.wm.decoder.terminal.forward(features)  # (H+1,TBI)

        self.wm.requires_grad_(True)
        return features, actions, rewards, terminals

class GCDreamerBehavior(ImagBehavior):
    '''
    Goal conditioned Dreamer behavior
    '''

    def __init__(self, conf, state_dim, world_model):
        super(GCDreamerBehavior, self).__init__(conf, world_model, True)

    def forward(self, feature, goal_embed):
        action_distr = self.ac.forward_actor(feature, goal_embed)  # (1,B,A)
        value = self.ac.forward_value(feature, goal_embed)  # (1,B)
        return action_distr, value

    def training_step(self, in_state_dream, H, goal_embed):
        return self.training_step(in_state_dream, H, self._cosine_similarity, goal_embed)

    # Reward computation using cosine similarity in feature space
    def _cosine_similarity(self, features, goal_embed):
        # goal_embed is the embedding of the fixed goal (E,)
        with torch.no_grad():
            batch_size = features.shape[0] * features.shape[1]
            init_state = self.init_state(batch_size)                                                # ((H+1)*TBI,D+S)
            action = torch.zeros(batch_size, self.conf.action_dim).to(self.device)                  # ((H+1)*TBI,A)
            reset_mask = torch.zeros(batch_size, 1).to(self.device)                                 # ((H+1)*TBI,1)
            batch_goal_embed = goal_embed.repeat(batch_size, 1)                                     # ((H+1)*TBI,E)
            _, (h, z) = self.wm.core.cell.forward(batch_goal_embed, action, reset_mask, init_state)
            goal_features = self.wm.core.to_feature(h, z).reshape(*features.shape)                  # ((H+1)*TBI,D+S)
        rewards = F.cosine_similarity(goal_features, features, dim=-1)                              # (H+1,TBI)

class Plan2Explore(ImagBehavior):
    '''
    Defines an ensemble of models used for exploration
    '''
    def __init__(self, conf, world_model):
        super(Plan2Explore, self).__init__(conf, world_model, False)
        self._conf = conf
        size = {
            'embed': 32 * conf.cnn_depth,
            'stoch': conf.stoch_dim * conf.stoch_discrete,
            'deter': conf.deter_dim,
            'feat': conf.stoch_dim * conf.stoch_discrete + conf.deter_dim,
        }[self._conf.disag_target]
        kw = dict(
            shape=size, layers=conf.disag_layers, units=conf.disag_units,
            act=conf.act
        )
        self.ensemble = nn.ModuleList([DenseHead(**kw) for _ in range(conf.disag_models)])

    def forward(self, features):
        action_distr = self.ac.forward_actor(features)  # (1,B,A)
        value = self.ac.forward_value(features)  # (1,B)
        return action_distr, value

    # if targets are not provided, then we assume we are just logging
    def training_step(self, in_state_dream, H, targets=None):
        # TODO: do the models in the ensemble not take action input?
        # Recall that in_state_dream has shape (h, z) = ((TBI,D), (TBI,S)) 
        # TODO: reshape in_state_dream to separate time dimension
        if targets:
            self._train_ensemble(in_state_dream, targets)
        return self.training_step(in_state_dream, H, self._intrinsic_reward)

    # Reward computation using disagreement in feature space
    def _intrinsic_reward(self, features):
        pred = torch.stack([head(features) for head in self.ensemble], dim=0)
        variance = torch.var(pred, dim=0)
        disagreement = torch.mean(variance)
        reward = self._conf.expl_intr_scale * disagreement
        return reward

    def _train_ensemble(self, inputs, targets):
        if self._conf.disag_offset:
            targets = targets[:, self._conf.disag_offset:]
            inputs = inputs[:, :-self._conf.disag_offset]
        preds = torch.stack([head(inputs) for head in self.ensemble], dim=0)
        likes = [torch.mean(preds.log_prob(targets), dim=-1)]
        loss = -torch.sum(likes)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return {'model_loss': loss.item()}