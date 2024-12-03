from self_play_trainer import Trainer as SelfPlayTrainer
import gym
from agents.agent_random import Player as RandomPlayer
from agents.agent_dqn import Player as DQNPlayer

env_name = 'neuron_poker-v0'
env = gym.make(env_name, initial_stacks=stack)
#env.add_player(PlayerShell(name='torch-rl', stack_size=5))
env.add_player(DQNPlayer())
env.add_player(DQNPlayer())
env.add_player(DQNPlayer())
trainer = SelfPlayTrainer(env)