import gym
from agents.agent_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
from random_trainer import Trainer as SelfPlayTrainer

env_name = 'neuron_poker-v0'
stack=5

env = gym.make(env_name, initial_stacks=stack, funds_plot=False, render=False)

env.add_player(DQNPlayer())
env.add_player(RandomPlayer())
env.add_player(RandomPlayer())

env.reset()

env.players[0].agent_obj.mount(env)
#for player in env.players:
#    player.agent_obj.mount(env)

#for player in env.players[1:]:
#    player.agent_obj.epsilon = 0.0

trainer = SelfPlayTrainer(env, batch_size=8, n_epochs=10, N=100, K=5)
trainer.run()