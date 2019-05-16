import argparse
import os
import types

import numpy as np
import torch
from helpers.vec_env.vec_normalize import VecNormalize
from models.factory import create_policy
from envs import make_vec_envs


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-path', default='',
                    help='path to checkpoint file')
parser.add_argument('--recurrent-policy', action='store_true', default=False,
                    help='use a recurrent policy')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--no-norm', action='store_true', default=False,
                    help='disables normalization')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
parser.add_argument('--no-realtime', action='store_true', default=False,
                    help='disables realtime mode and rendering for obt env')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.realtime = not args.no_realtime

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

num_env = 1
env = make_vec_envs(args.env_name, args.seed + 1000,
                    num_env, gamma=None, no_norm=args.no_norm,
                    num_stack=args.num_stack, log_dir=None, add_timestep=args.add_timestep,
                    device=device, eval=True, allow_early_resets=False, realtime=args.realtime)

# Get a render function
render_func = None
tmp_env = env
while True:
    if hasattr(tmp_env, 'envs'):
        render_func = tmp_env.envs[0].render
        break
    elif hasattr(tmp_env, 'venv'):
        tmp_env = tmp_env.venv
    elif hasattr(tmp_env, 'env'):
        tmp_env = tmp_env.env
    else:
        break

# We need to use the same statistics for normalization as used in training
state_dict, ob_rms = torch.load(args.load_path)

noisy_net = True
actor_critic = create_policy(
    env.observation_space,
    env.action_space,
    name='basic',
    nn_kwargs={
        # 'batch_norm': False if args.algo == 'acktr' else True,
        'recurrent': 'lstm' if args.recurrent_policy else '',
        'hidden_size': 512,
    },
    noisy_net=noisy_net,
    train=False)

actor_critic.load_state_dict(state_dict)
actor_critic.to(device)
actor_critic.eval()

recurrent_hidden_states = torch.zeros(2, num_env, actor_critic.recurrent_hidden_state_size).to(device)
masks = torch.zeros(num_env, 1).to(device)

obs = env.reset()

if render_func is not None:
    render_func('human') # NOTE obstacle tower env uses realtime flag instead

episode_reward = [0.] * num_env
while True:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=True)

    obs, reward, done, _ = env.step(action)
    for i, r in enumerate(reward):
        episode_reward[i] += r

    masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

    for i, d in enumerate(done):
        if d:
            print('Episode reward for env:%d = %f' % (i, episode_reward[i]))
            episode_reward[i] = 0.

    if render_func is not None:
        render_func('human')
