from agents.agent_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
import gym
import numpy as np
from gym_env.enums import Action, Stage

env_name = 'neuron_poker-v0'
stack=5

env = gym.make(env_name, initial_stacks=stack, funds_plot=False, render=False)

env.add_player(DQNPlayer())
env.add_player(RandomPlayer())
env.add_player(RandomPlayer())

env.reset()
env.players[0].agent_obj.mount(env)
env.players[0].agent_obj.epsilon = 0.5

state = env.reset()
done = False
print(Action(Action.FOLD))


while not env.done:
        #print(env.current_player.name)
        legal_moves = env.legal_moves
        observation = env.observation
        #print(legal_moves)
        action = env.current_player.agent_obj.action(legal_moves, observation, None)
        #print(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        

print('done')
print(env.ep_winner_idx)