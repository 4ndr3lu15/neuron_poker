import gym
from agents.agent_dqn import Player as DQNPlayer
from self_play_trainer import Trainer as SelfPlayTrainer

env_name = 'neuron_poker-v0'
stack=5

env = gym.make(env_name, initial_stacks=stack, funds_plot=False, render=False)

env.add_player(DQNPlayer())
env.add_player(DQNPlayer())
env.add_player(DQNPlayer())

env.reset()

for player in env.players:
    player.agent_obj.mount(env)

trainer = SelfPlayTrainer(env)
trainer.run()