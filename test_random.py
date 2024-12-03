from agents.agent_torch_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
from gym_env.env import PlayerShell
import gym
import numpy as np

env_name = 'neuron_poker-v0'
env = gym.make(env_name, initial_stacks=5, funds_plot=False, render=False)

np.random.seed(123)
env.seed(123)

env.add_player(RandomPlayer())
env.add_player(RandomPlayer())
#env.add_player(RandomPlayer())
# env.add_player(PlayerShell(name='keras-rl', stack_size=self.stack))  # shell is used for callback to keras rl
#env.add_player(PlayerShell(name='torch-rl', stack_size=5)) 
#env.add_player(PlayerShell(name='torch-rl', stack_size=5)) 
env.add_player(PlayerShell(name='torch-rl', stack_size=5)) 

env.setup()

dqn = DQNPlayer(env=env)
# dqn.initiate_agent(env)
dqn.train(env_name='torch-rl')