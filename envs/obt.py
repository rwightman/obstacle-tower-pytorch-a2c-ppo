import torch
import numpy as np
import gym
from obstacle_tower_env import ObstacleTowerEnv


class ToTorchTensors(gym.ObservationWrapper):
    def __init__(self, env=None, device='cpu'):
        super(ToTorchTensors, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)
        self.device = device

    def observation(self, observation):
        tensor = torch.from_numpy(np.rollaxis(observation, 2)).to(self.device)
        tensor = 2 * (tensor.float() / 255 - 0.5)
        return tensor


def make_env(env_id, rank, env_filename='./ObstacleTower/obstacletower', docker_training=False):
    env = ObstacleTowerEnv(env_filename, docker_training=docker_training, worker_id=rank)
    return env
