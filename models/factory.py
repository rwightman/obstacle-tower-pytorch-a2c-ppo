from .model_pomm import PommNet
from .model_generic import CNNBase, MLPBase
from .policy import Policy


def create_policy(obs_space, action_space, name='basic', nn_kwargs={}, train=True, noisy_net=False):
    nn = None
    obs_shape = obs_space.shape
    if name.lower() == 'basic':
        if len(obs_shape) == 3:
            print('Creating CNN policy')
            nn = CNNBase(obs_shape[0], noisy_net=noisy_net, **nn_kwargs)
        elif len(obs_shape) == 1:
            print('Creating MLP policy')
            nn = MLPBase(obs_shape[0], **nn_kwargs)
        else:
            raise NotImplementedError
    elif name.lower() == 'pomm':
        nn = PommNet(obs_shape=obs_shape, **nn_kwargs)
    else:
        assert False and "Invalid policy name"

    if train:
        nn.train()
    else:
        nn.eval()

    policy = Policy(nn, action_space=action_space, noisy_net=noisy_net)

    return policy
