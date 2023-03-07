import torch
import torch.nn as nn

from .a2c import *
from .functions import *
from .common import *

class ImagBehavior(nn.Module):

    def __init__(self, conf, world_model, use_gc=False):
        super().__init__()
        self.conf = conf
        # TODO: we may want to add a separate config argument for actor_grad for explorer/achiever actors
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

    def train(self, in_state_dream, H, reward_func, goal_embed=None):
        # Note features_dream includes the starting "real" features at features_dream[0]
        features_dream, actions_dream, rewards_dream, terminals_dream = \
            self.dream(in_state_dream, H, reward_func, self.conf.actor_grad == 'dynamics', goal_embed)  # (H+1,TBI,D)

        (loss_actor, loss_critic), metrics_ac, tensors_ac = \
            self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream, goal_embed)
        return loss_actor, loss_critic, metrics_ac, features_dream, actions_dream, rewards_dream, terminals_dream, tensors_ac

    # unrolls a policy in imagination using the world model
    def dream(self, in_state: StateB, imag_horizon: int, reward_func, dynamics_gradients=False, goal_embed=None):
        features = []
        actions = []
        priors = []
        state = in_state
        self.wm.requires_grad_(False)  # Prevent dynamics gradients from affecting world model

        for i in range(imag_horizon):
            feature = self.wm.core.to_feature(*state)
            if goal_embed is None:
                action_dist = self.ac.forward_actor(feature)
            else:
                action_dist = self.ac.forward_actor(feature, goal_embed)
            if dynamics_gradients:
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
            features.append(feature)
            actions.append(action)
            # When using dynamics gradients, this causes gradients in RSSM, which we don't want.
            # This is handled in backprop - the optimizer_model will ignore gradients from loss_actor.
            prior, state = self.wm.core.cell.forward_prior(action, None, state)
            priors.append(prior)

        # save one extra feature for last terminal prediction
        feature = self.wm.core.to_feature(*state)
        features.append(feature)

        if goal_embed is None:
            features = torch.stack(features[1:])  # (H,TBI,D)
            actions = torch.stack(actions[1:])  # (H-1,TBI,A)
            priors = torch.stack(priors[:-1])  # (H-1,TBI,2S)
            rewards = reward_func(priors, features[:-1], actions)               # (H-1,TBI)
        
        # deprecating achiever functionality for now since a2c impl differs
        # else:
        #     features = torch.stack(features)  # (H+1,TBI,D)
        #     actions = torch.stack(actions)  # (H,TBI,A)
        #     rewards = reward_func(features, goal_embed) # (H+1,TBI)
        terminals = self.wm.decoder.terminal.forward(features)  # (H,TBI)

        self.wm.requires_grad_(True)
        return features, actions, rewards, terminals

class GCDreamerBehavior(ImagBehavior):
    '''
    Goal conditioned Dreamer behavior
    '''

    def __init__(self, conf, world_model):
        super(GCDreamerBehavior, self).__init__(conf, world_model, True)

    def forward(self, feature, goal_embed, forward_only=False):
        action_distr = self.ac.forward_actor(feature, goal_embed)  # (1,B,A)
        value = self.ac.forward_value(feature, goal_embed)  # (1,B)
        return action_distr, value

    def training_step(self, in_state_dream, H, goal_embed):
        return self.train(in_state_dream, H, self._cosine_similarity, goal_embed)

    # Reward computation using cosine similarity in feature space
    def _cosine_similarity(self, features, goal_embed):
        # goal_embed is the embedding of the fixed goal (E,)
        device = next(self.wm.parameters()).device
        with torch.no_grad():
            batch_size = features.shape[0] * features.shape[1]
            init_state = self.wm.init_state(batch_size)                                                # ((H+1)*TBI,D+S)
            action = torch.zeros(batch_size, self.conf.action_dim).to(device)                  # ((H+1)*TBI,A)
            reset_mask = torch.zeros(batch_size, 1).to(device)                                 # ((H+1)*TBI,1)
            batch_goal_embed = goal_embed.repeat(batch_size, 1)                                     # ((H+1)*TBI,E)
            _, (h, z) = self.wm.core.cell.forward(batch_goal_embed, action, reset_mask, init_state)
            goal_features = self.wm.core.to_feature(h, z).reshape(*features.shape)                  # ((H+1)*TBI,D+S)
        rewards = F.cosine_similarity(goal_features, features, dim=-1)                              # (H+1,TBI)
        return rewards

class Plan2Explore(ImagBehavior):
    '''
    Defines an ensemble of models used for exploration
    '''
    def __init__(self, conf, world_model):
        super(Plan2Explore, self).__init__(conf, world_model, False)
        self.conf = conf
        out_size = {
            'embed': 32 * conf.cnn_depth,
            'stoch': conf.stoch_dim * conf.stoch_discrete,
            'deter': conf.deter_dim,
            'feat': conf.stoch_dim * conf.stoch_discrete + conf.deter_dim,
        }[self.conf.disag_target]
        in_size = conf.deter_dim + out_size + conf.action_dim
        kw = dict(in_size=in_size, out_size=out_size, layers=conf.disag_layers, units=conf.disag_units)
        self.ensemble = nn.ModuleList([DenseHead(**kw) for _ in range(conf.disag_models)])

    def forward(self, features):
        action_distr = self.ac.forward_actor(features)  # (1,B,A)
        value = self.ac.forward_value(features)  # (1,B)
        return action_distr, value

    # if targets are not provided, then we assume we are just logging
    # we no longer need posts, because we are only interested in tracking actions that affect our rewards
    def training_step(self, in_state_dream, H, posts, actions, features, forward_only=False):
        if forward_only: # for logging
            return self.dream(in_state_dream, H, lambda *x: torch.eye(1), self.conf.actor_grad == 'dynamics', None)
        ensemble_loss = self._train_ensemble(posts, actions, features)
        return ensemble_loss, *self.train(in_state_dream, H, self._intrinsic_reward)

    # Reward computation using disagreement in the space of the prior predictions
    def _intrinsic_reward(self, priors, features, actions):
        # posts (TBI,2S), priors (H,TBI,2S)
        self.ensemble.requires_grad_(False)  # Prevent dynamics gradients from affecting ensemble
        # priors = torch.cat([posts.unsqueeze(0), priors], dim=0) # (H+1,TBI,2S)
        pred = torch.stack([head(priors, features, actions).mean for head in self.ensemble], dim=0) # (K,H+1,TBI,2S)
        variance = torch.var(pred, dim=0) # (H+1,TBI,2S)
        disagreement = torch.mean(variance, dim=-1) # (H+1,TBI)
        reward = self.conf.expl_intr_scale * disagreement
        self.ensemble.requires_grad_(True)
        return reward # (H+1,TBI)

    # TODO: I didn't even think about how this would be trained in the new setting
    # Also note: the only reason we aren't using the extra initial state is because it will complicate things
    def _train_ensemble(self, posts, actions, features):
        # posts are shape (T,B,I,2S) where I=1
        # we have the action preceeding every post
        # features and actions are shape (T,B,I,D) and (T,B,I,A)
        actions = actions[1:] # (T-1,B,I,A)
        features = features[:-1] # (T-1,B,I,D)
        if self.conf.disag_offset:
            target_posts = posts[self.conf.disag_offset:]
            input_posts = posts[:-self.conf.disag_offset]
        actions, _ = flatten_batch(actions) # (T,B,I,2S) => (TBI,2S)
        features, _ = flatten_batch(features) # (T,B,I,2S) => (TBI,2S)
        input_posts, _ = flatten_batch(input_posts) # (T,B,I,2S) => (TBI,2S)
        target_posts, _ = flatten_batch(target_posts) # (T,B,I,2S) => (TBI,2S)
        preds = [head(input_posts, actions, features) for head in self.ensemble] # (K,TBI,2S)
        likes = torch.stack([torch.mean(pred.log_prob(target_posts), dim=-1) for pred in preds]) # (K,)
        loss = -torch.sum(likes)
        return loss