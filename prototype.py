#import gym_env
import gym
from agents.agent_random import Player as RandomPlayer
from agents.agent_consider_equity import Player as EquityPlayer
import pandas as pd

env_name = 'neuron_poker-v0'
num_of_plrs = 3
stack=20
render = False

env_name = 'neuron_poker-v0'
env = gym.make(env_name, initial_stacks=stack, render=render)
#env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
#env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
#env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
#env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
#env.add_player(RandomPlayer())

for _ in range(num_of_plrs):
    env.add_player(RandomPlayer())

print(env.acting_agent)
env.step()