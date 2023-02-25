# Ignore annoying warnings from imported envs
import warnings
warnings.filterwarnings("ignore", ".*Box bound precision lowered by casting")  # gym

import gym
import numpy as np

# from .import minecraft_minerl
from .wrappers import *


def create_env(env_id: str, no_terminal: bool, env_time_limit: int, env_action_repeat: int, worker_id: int):

    if env_id.startswith('MiniGrid-'):
        from .minigrid import MiniGrid
        env = MiniGrid(env_id)

    elif env_id.startswith('Atari-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)

    elif env_id.startswith('AtariGray-'):
        from .atari import Atari
        env = Atari(env_id.split('-')[1].lower(), action_repeat=env_action_repeat, grayscale=True)

    elif env_id.startswith('MiniWorld-'):
        import gym_miniworld.wrappers as wrap
        env = gym.make(env_id)
        env = wrap.DictWrapper(env)
        env = wrap.MapWrapper(env)
        env = wrap.AgentPosWrapper(env)
        if env_id.startswith('MiniWorld-ScavengerHunt'):
            env = wrap.GoalPosWrapper(env)
            env = wrap.GoalVisibleWrapper(env)
            env = wrap.GoalVisAgeWrapper(env)

    elif env_id.startswith('DmLab-'):
        from .dmlab import DmLab
        env = DmLab(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat)
        env = DictWrapper(env)
    
    elif env_id.startswith('DMM-'):
        from .dmm import DMMEnv
        env = DMMEnv(env_id.split('-', maxsplit=1)[1].lower(), num_action_repeats=env_action_repeat, worker_id=worker_id)
        env = DictWrapper(env)

    elif env_id.startswith('MineRL'):
        from .minerl import MineRL
        constr = lambda: MineRL(env_id, action_repeat=env_action_repeat)
        env = RestartOnExceptionWrapper(constr)

    elif env_id.startswith('DMC-'):
        from .dmc import DMC
        env = DMC(env_id.split('-')[1].lower(), action_repeat=env_action_repeat)

    elif env_id.startswith('crafter'):
        # I don't think we need to have crafter in envs, just init from crafter package
        import crafter
        env = gym.make('CrafterReward-v1') # Or CrafterNoReward-v1
        # can add recorder wrapper here for videos
        env = DictWrapper(env) # other gym envs use this wrapper

    # elif env_id.startswith('minecraft'):
    #     env = minecraft_minerl.MineRLEnv().make()
    #     env = DictWrapper(env) # other gym envs use this wrapper

    else:
        env = gym.make(env_id)
        env = DictWrapper(env)

    # if the env is a subclass of Embodied.Env, we should use the logic from dreamerv3
    # TODO: do we need embodied for minecraft env?
    if hasattr(env.action_space, 'n'):
        env = OneHotActionWrapper(env)
    if env_time_limit > 0:
        env = TimeLimitWrapper(env, env_time_limit)
    env = ActionRewardResetWrapper(env, no_terminal)
    env = CollectWrapper(env)
    return env
