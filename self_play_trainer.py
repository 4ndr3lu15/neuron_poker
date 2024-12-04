import random
from collections import deque
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, env, batch_size=64, n_epochs=3, N=5, K=2, ß=3, æ=0.5):
        self.env = env
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.N = N # number of episodes per epoch
        self.K = K # number of episodes between internal updates
        self.ß = ß # number of episodes for evaluation
        self.æ = æ # winrate threshold for copying weights

        self.gamma = 0.99
        self.optimizer = torch.optim.Adam(self.env.players[0].agent_obj.policy_net.parameters(), lr=0.001)

        self.current_epoch = 0
        self.current_train_ep = 0
        self.current_train_step = 0

        self.writer = SummaryWriter()

    def run(self):
        """Run the training loop"""
        print(f"Training for {self.n_epochs} epochs - {self.N} train episodes and {self.ß} eval episodes per epoch")
        for _ in range(self.n_epochs):
            self.train_one_epoch()
            self.save_model()    
            self.current_epoch += 1

    def train_one_epoch(self):
        """Train the agent for one epoch"""
        print(f"Training epoch {self.current_epoch}")
        self.current_epoch_ep = 0
        self.run_train_eps()
        ç = self.run_eval_eps()

        if ç > self.æ:
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

        print(f"Running {mode} episode {self.current_epoch_ep} from epoch {self.current_epoch}")

        state = self.env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not self.env.done:

            legal_moves = self.env.legal_moves
            observation = self.env.observation
            action = self.env.current_player.agent_obj.action(legal_moves, observation, None)
            next_state, reward, done, _ = self.env.step(action)

            if mode == 'train':
                self.replay_buffer.append((state, action, reward, next_state, done))
                episode_reward += reward
                
            state = next_state
            steps += 1
        

        self.current_epoch_ep += 1

        print(f"Episode reward: {episode_reward} - Steps: {steps} - Winner: {self.env.ep_winner_idx}")

        if mode == 'train':

            self.optimize()

            self.writer.add_scalar('Episode Reward', episode_reward, self.current_train_ep)
            self.writer.add_scalar('Steps per Episode', steps, self.current_train_ep)
            self.writer.add_scalar('Winning Player', self.env.ep_winner_idx, self.current_train_ep)

            self.current_train_ep += 1

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
            x = self.run_episode(mode='eval')
            counters[x] += 1
        return counters[0] / self.ß

    def internal_update(self):
        self.env.players[0].agent_obj.target_net.load_state_dict(self.env.players[0].agent_obj.policy_net.state_dict())

    def external_update(self):
        for player in self.env.players[1:]:
            player.agent_obj.target_net.load_state_dict(self.env.players[0].agent_obj.policy_net.state_dict())

    def save_model(self):
        path = f"models/{self.current_epoch}.pth"
        torch.save(self.env.players[0].agent_obj.policy_net.state_dict(), path)
