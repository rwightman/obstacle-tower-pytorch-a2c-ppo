# PyTorch A2C/PPO for Obstacle Tower Challenge

Adapted from my A2C/PPO experiments for Pommerman, this was the basis for some experiments with actor-critic PG algorithms for the Obstacle Tower Challenge (https://github.com/Unity-Technologies/obstacle-tower-challenge)

The reinforcement learning codebase is based upon Ilya Kostrikov's awesome work (https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)

## Changes

In short:
* Add Noisy Networks to replace entropy regularization for exploration
* Replace GRU RNN policy with LSTM

My initial attempts with A2C and PPO went nowhere. After training for days, average floor remained at 1 with little progress made by either algo. Seeing the success of my experiments with Rainbow (https://github.com/rwightman/obstacle-towers-pytorch-rainbow), I decided to bring in Noisy Networks from (https://github.com/Kaixhin/Rainbow) to improve exploration. Additionally I replaced the GRU with LSTM.

This resulted in some progress. Floor average around 6-7 was reached, not quite as good as my Rainbow experiments. Top floor of 9-10 was hit on occasion but also a lot of failures to move past floor 2 with the same policy. 

Minimal time was spent searching the hyper-param space, I'm sure much better could be achieved.  


## Usage

* Setup a new Conda Python 3.6 environment (do not use 3.7! compatibility issues with Unity's support modules)
* Install recent (ver 1.x) of PyTorch
* Setup environment download engine as per: https://github.com/Unity-Technologies/obstacle-tower-challenge#local-setup-for-training but using this repo in place of that clone and do it within the same Conda env
* Run `python main.py --env-name obt --algo ppo --use-gae --recurrent-policy --num-processes 32 --num-mini-batch 8 --num-steps 120 --entropy-coef 0.0 --lr 1e-4 --clip-param 0.1` and wait...
* enjoy.py can be used to watch the trained policy in realtime 
