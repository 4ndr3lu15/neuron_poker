from agents.agent_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
import gym
from get_latest_model import get_latest_model

env_name = 'neuron_poker-v0'
stack=20

env = gym.make(env_name, initial_stacks=stack, funds_plot=False, render=False)

env.add_player(DQNPlayer())
env.add_player(RandomPlayer())
env.add_player(RandomPlayer())

env.reset()
#env.players[0].agent_obj.mount(env)

model_path = get_latest_model("models")
env.players[0].agent_obj.mount(env, model_path)
env.players[0].agent_obj.epsilon = 0.0

def run_ep(env):
    
    env.reset()
    steps = 0
    
    while not env.done:
        legal_moves = env.legal_moves
        observation = env.observation
        action = env.current_player.agent_obj.action(legal_moves, observation, None)
        _, _, _, _ = env.step(action)
        steps += 1

    return env.ep_winner_idx, steps


kek = [0, 0, 0]


for _ in range(100):

        print(f"running ep {_}")
        winner_idx, steps = run_ep(env)
        kek[winner_idx] += 1
        print(f"winner: {winner_idx}, steps: {steps}")


print(kek)
print(sum(kek))
winrate = kek[0] / sum(kek)
print(f'winrate: {winrate*100}%')