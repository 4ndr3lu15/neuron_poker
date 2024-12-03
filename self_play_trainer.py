#import gym_env
import gym
from agents.agent_random import Player as RandomPlayer
from agents.agent_torch_dqn import Player as DQNPlayer
#from agents.agent_torch_dqn_fixed import Player as DQNPlayerFixed
import pandas as pd
import random
from gym_env.enums import Action
from collections import deque
import torch
import numpy as np
import torch.nn as nn


class Trainer:

    def __init__(self, env, batch_size=64, n_epochs=10, N=5, K=2, ß=10):
        self.env = env
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.N = N # number of episodes per epoch
        self.K = K # number of episodes between internal updates
        self.ß = ß # number of episodes for evaluation

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = torch.optim.Adam(self.env.players[0].agent_obj.policy_net.parameters(), lr=0.001)

        self.current_epoch = 0

    def run(self):
        """Run the training loop"""
        print(f"Training for {self.n_epochs} epochs - {self.N} episodes each")
        for _ in range(self.n_epochs):
            self.train_one_epoch()    

    def train_one_epoch(self):
        """Train the agent for one epoch"""
        print(f"Training epoch {self.current_epoch}")
        self.current_ep = 0
        self.current_epoch += 1
        self.run_train_eps()
        ç = self.run_eval_eps()
        if ç > 0.40:
            print("Winrate is good, copying weights")
            self.external_update()
        else:
            print("Winrate is too low, not copying weights")

    def run_train_eps(self):
        """Run N episodes, store experiences in replay buffer"""
        print(f"Running {self.N} training episodes")
        for episode in range(self.N):
            self.run_episode(mode='train')
            if episode % self.K == 0:
                self.internal_update()

    def run_episode(self, mode=None):
        """Run a single episode"""
        print(f"Running episode {self.current_ep} - mode: {mode}")
        state = self.env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not self.env.done:
            legal_moves = self.env.legal_moves
            observation = self.env.observation
            action = self.env.current_player.agent_obj.action(legal_moves, observation, None)
            next_state, reward, done, _ = self.env.step(action)
            if mode == 'train': # and self.env.current_player == self.env.players[0]:
                self.replay_buffer.append((state, action, reward, next_state, done))
            elif self.env.current_player == self.env.players[0]:
                episode_reward += reward
            state = next_state
            steps += 1
        
        self.current_ep += 1
        print(f"Episode reward: {episode_reward} - Steps: {steps}")
        print(f"Winner: {self.env.ep_winner_idx}")

        if mode == 'train':
            self.optimize()

        return self.env.ep_winner_idx

    def optimize(self):
        """Train the agent with a batch of experiences"""

        print("Optimizing policy network")
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        # print(type(actions), actions)
        # actions = torch.LongTensor(actions).unsqueeze(1)
        actions = torch.LongTensor([action.value if type(action) != np.int64 else action for action in actions ]).unsqueeze(1)
        # print('actions converted to tensor')
        # logger.info(f'actions: {actions}')
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q values for current states (policy net)
        current_q_values = self.env.players[0].agent_obj.policy_net(states).gather(1, actions).squeeze()

        # Q values for next states (target net)
        next_q_values = self.env.players[0].agent_obj.target_net(next_states).max(1)[0]
        next_q_values[dones == 1] = 0.0  # Set next q values to 0 where episode ended

        # Compute target
        target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.env.players[0].agent_obj.epsilon > self.env.players[0].agent_obj.epsilon_min:
            self.env.players[0].agent_obj.epsilon *= self.env.players[0].agent_obj.epsilon_decay

    def run_eval_eps(self):
        """Run N episodes, return the winrate of trainable agent"""
        counters = {0: 0, 1: 0, 2: 0}
        for _ in range(self.ß):
            x = self.run_episode()
            counters[x] += 1
        return counters[0] / self.ß

    def internal_update(self):
        self.env.players[0].agent_obj.target_net.load_state_dict(self.env.players[0].agent_obj.policy_net.state_dict())

    def external_update(self):
        for player in self.env.players[1:]:
            player.agent_obj.target_net.load_state_dict(self.env.players[0].agent_obj.policy_net.state_dict())
