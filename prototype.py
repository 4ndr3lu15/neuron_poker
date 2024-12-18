#import gym_env
import gym
from agents.agent_random import Player as RandomPlayer
from agents.agent_dqn import Player as DQNPlayer
#from agents.agent_torch_dqn_fixed import Player as DQNPlayerFixed
from gym_env.env import PlayerShell
import pandas as pd
import random
from gym_env.enums import Action
from self_play_trainer import Trainer as SelfPlayTrainer

env_name = 'neuron_poker-v0'
stack=5

env = gym.make(env_name, initial_stacks=stack)

env.add_player(DQNPlayer())
env.add_player(DQNPlayer())
env.add_player(DQNPlayer())

env.reset()

for player in env.players:
    player.agent_obj.mount(env)

trainer = SelfPlayTrainer(env)
trainer.run()