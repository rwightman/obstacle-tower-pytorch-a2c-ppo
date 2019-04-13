import torch
import numpy as np
import gym
from obstacle_tower_env import ObstacleTowerEnv


def make_env(env_id, rank, env_filename='./ObstacleTower/obstacletower', docker_training=False):
    env = ObstacleTowerEnv(env_filename, docker_training=docker_training, worker_id=rank)
    return env
