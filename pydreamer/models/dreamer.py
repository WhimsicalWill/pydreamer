from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..tools import *
from .a2c import *
from .common import *
from .functions import *
from .encoders import *
from .decoders import *
from .rnn import *
from .rssm import *
from .probes import *
from .exploration import *
from .behaviors import *


class Dreamer(nn.Module):

    def __init__(self, conf):
        super().__init__()
        assert conf.action_dim > 0, "Need to set action_dim to match environment"
        state_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)

        # World model

        self.wm = WorldModel(conf)

        # Pass World Model into task_behavior and explore_behavior

        # TODO: change arguments?
        self._task_behavior = GCDreamerBehavior(conf, state_dim, self.wm)
        self._expl_behavior = Plan2Explore(conf, self.wm)

        # Map probe

        if conf.probe_model == 'map':
            probe_model = MapProbeHead(state_dim + 4, conf)
        elif conf.probe_model == 'goals':
            probe_model = GoalsProbe(state_dim, conf)
        elif conf.probe_model == 'none':
            probe_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
        optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
        optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=lr_actor or lr, eps=eps)
        optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=lr_critic or lr, eps=eps)
        return optimizer_wm, optimizer_probe, optimizer_actor, optimizer_critic

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        grad_metrics = {
            'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
            'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip),
            'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), grad_clip_ac or grad_clip),
            'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), grad_clip_ac or grad_clip),
        }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any,
                behavior='achiever',
                ) -> Tuple[D.Distribution, Any, Dict]:
        assert 'action' in obs, 'Observation should contain previous action'
        act_shape = obs['action'].shape
        assert len(act_shape) == 3 and act_shape[0] == 1, f'Expected shape (1,B,A), got {act_shape}'

        # Forward (world model)

        features, out_state = self.wm.forward(obs, in_state)

        # Forward (actor critic)

        feature = features[:, :, 0]  # (T=1,B,I=1,F) => (1,B,F)

        if behavior == 'achiever':
            action_distr, value = self._task_behavior(features)
        else:
            action_distr, value = self._expl_behavior(features)


        metrics = dict(policy_value=value.detach().mean())
        return action_distr, out_state, metrics

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      imag_horizon: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      do_dream_tensors=False,
                      ):
        assert 'action' in obs, '`action` required in observation'
        assert 'reward' in obs, '`reward` required in observation'
        assert 'reset' in obs, '`reset` required in observation'
        assert 'terminal' in obs, '`terminal` required in observation'
        T, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon

        # World model

        loss_model, features, states, out_state, metrics, tensors = \
            self.wm.training_step(obs,
                                  in_state,
                                  iwae_samples=iwae_samples,
                                  do_open_loop=do_open_loop,
                                  do_image_pred=do_image_pred)

        # Map probe

        loss_probe, metrics_probe, tensors_probe = self.probe_model.training_step(features.detach(), obs)
        metrics.update(**metrics_probe)
        tensors.update(**tensors_probe)

        # TODO: should this be copied if it's used twice?
        in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])  # type: ignore  # (T,B,I) => (TBI)

        # Explore Behavior Training Step (explorer)

        loss_actor, loss_critic = self._expl_behavior.training_step(features, obs, in_state_dream, iwae_samples, imag_horizon)
        
        # Task Behavior Training Step (achiever)
        
        loss_actor, loss_critic = self._task_behavior.training_step(features, obs, in_state_dream, iwae_samples, imag_horizon)

        # Dream for a log sample.

        dream_tensors = {}
        if do_dream_tensors and self.wm.decoder.image is not None:
            with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
                # The reason we don't just take real features_dream is because it's really big (H*T*B*I),
                # and here for inspection purposes we only dream from first step, so it's (H*B).
                # Oh, and we set here H=T-1, so we get (T,B), and the dreamed experience aligns with actual.
                in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (T,B,I) => (B)
                features_dream, actions_dream, rewards_dream, terminals_dream = self.dream(in_state_dream, T - 1)  # H = T-1
                image_dream = self.wm.decoder.image.forward(features_dream)
                _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream, log_only=True)
                # The tensors are intentionally named same as in tensors, so the logged npz looks the same for dreamed or not
                dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
                                     reward_pred=rewards_dream.mean,
                                     terminal_pred=terminals_dream.mean,
                                     image_pred=image_dream,
                                     **tensors_ac)
                assert dream_tensors['action_pred'].shape == obs['action'].shape
                assert dream_tensors['image_pred'].shape == obs['image'].shape

        # TODO: modify metrics, tensors so that we have copies for explorer and achiever
        # TODO: also determine how metrics are logged and what they're used for
        return (loss_model, loss_probe, loss_actor, loss_critic), out_state, metrics, tensors, dream_tensors

    def __str__(self):
        # Short representation
        s = []
        s.append(f'Model: {param_count(self)} parameters')
        for submodel in (self.wm.encoder, self.wm.decoder, self.wm.core, self.ac, self.probe_model):
            if submodel is not None:
                s.append(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
        return '\n'.join(s)

    def __repr__(self):
        # Long representation
        return super().__repr__()


class WorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()

        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.kl_balance = None if conf.kl_balance == 0.5 else conf.kl_balance

        # Encoder

        self.encoder = MultiEncoder(conf)

        # Decoders

        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.decoder = MultiDecoder(features_dim, conf)

        # RSSM

        self.core = RSSMCore(embed_dim=self.encoder.out_dim,
                             action_dim=conf.action_dim,
                             deter_dim=conf.deter_dim,
                             stoch_dim=conf.stoch_dim,
                             stoch_discrete=conf.stoch_discrete,
                             hidden_dim=conf.hidden_dim,
                             gru_layers=conf.gru_layers,
                             gru_type=conf.gru_type,
                             layer_norm=conf.layer_norm)

        # Init

        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self.core.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any
                ):
        loss, features, states, out_state, metrics, tensors = \
            self.training_step(obs, in_state, forward_only=True)
        return features, out_state

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      forward_only=False
                      ):

        # Encoder

        embed = self.encoder(obs)

        # RSSM

        prior, post, post_samples, features, states, out_state = \
            self.core.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)

        if forward_only:
            return torch.tensor(0.0), features, states, out_state, {}, {}

        # Decoder

        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)

        # KL loss

        d = self.core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B,I)
        if iwae_samples == 1:
            # Analytic KL loss, standard for VAE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)

        # Total loss

        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)

        # Metrics

        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl_exact, dim=2)  # Log exact KL loss even when using IWAE, it avoids random negative values
            entropy_prior = dprior.entropy().mean(dim=2)
            entropy_post = dpost.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(),
                           entropy_prior=entropy_prior,
                           entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(),
                           loss_kl=loss_kl.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean())

        # Predictions

        if do_image_pred:
            with torch.no_grad():
                prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
                features_prior = self.core.feature_replace_z(features, prior_samples)
                # Decode from prior
                _, mets, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
                metrics_logprob = {k.replace('loss_', 'logprob_'): v for k, v in mets.items() if k.startswith('loss_')}
                tensors_logprob = {k.replace('loss_', 'logprob_'): v for k, v in tens.items() if k.startswith('loss_')}
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                metrics.update(**metrics_logprob)   # logprob_image, ...
                tensors.update(**tensors_logprob)  # logprob_image, ...
                tensors.update(**tensors_pred)  # image_pred, ...

        return loss_model.mean(), features, states, out_state, metrics, tensors


class GCWorldModel(WorldModel):
    '''
    Goal conditioned World Model
    '''

    def __init__(self, conf):
        pass
