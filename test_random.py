from agents.agent_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
import gym
import numpy as np

env_name = 'neuron_poker-v0'
stack=20

env = gym.make(env_name, initial_stacks=stack, funds_plot=False, render=False)

env.add_player(DQNPlayer())
env.add_player(RandomPlayer())
env.add_player(RandomPlayer())

env.reset()
env.players[0].agent_obj.mount(env)

state = env.reset()
done = False

while not env.done:
    
        legal_moves = env.legal_moves
        observation = env.observation
        action = env.current_player.agent_obj.action(legal_moves, observation, None)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        

print('done')
print(env.ep_winner_idx)