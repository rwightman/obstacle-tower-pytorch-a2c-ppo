import torch
import torch.nn as nn

# Factorised NoisyLinear layer with bias
from models.noisy import create_linear


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        self.rnn = None
        if recurrent.lower() == 'gru':
            self.rnn = nn.GRU(recurrent_input_size, hidden_size)
        elif recurrent.lower() == 'lstm':
            self.rnn = nn.LSTM(recurrent_input_size, hidden_size)
        if self.rnn is not None:
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def recurrent_hidden_shape(self):
        if isinstance(self.rnn, nn.LSTM):
            return 2, self._hidden_size
        elif isinstance(self.rnn, nn.GRU):
            return 1, self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_rnn(self, x, hxs, masks):
        hxs, cxs = torch.chunk(hxs, 2, dim=0)
        if x.size(0) == hxs.size(1):
            hxs = (hxs * masks)
            cxs = (cxs * masks)
            x, (hxs, cxs) = self.rnn(x.unsqueeze(0), (hxs, cxs))
            x = x.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(1)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            #hxs = hxs.unsqueeze(0)
            #cxs = cxs.unsqueeze(0)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                hmask = masks[start_idx].view(1, -1, 1)
                hxs = hxs * hmask
                cxs = cxs * hmask

                rnn_scores, (hxs, cxs) = self.rnn(
                    x[start_idx:end_idx],
                    (hxs, cxs)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        hxs = torch.stack((hxs.squeeze(0), cxs.squeeze(0)), dim=0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512, noisy_net=False):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)
        self.noisy = noisy_net

        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            create_linear(64 * 7 * 7, hidden_size, noisy=noisy_net, std=0.1),
            nn.ReLU()
        )

        self.critic_linear = create_linear(hidden_size, 1, noisy=noisy_net, std=0.1)

    def forward(self, inputs, rnn_hxs, masks):
        x = (inputs/127.5 - 1.).float()
        x = self.main(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

    def reset_noise(self):
        self.main[7].reset_noise()
        self.critic_linear.reset_noise()


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        self.critic_linear = nn.Linear(hidden_size, 1)

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
