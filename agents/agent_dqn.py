import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import json
import time
from gym_env.enums import Action, Stage
import logging
from neural_nets.dqn_nn import DQNetwork

window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 100000
memory_limit = int(nb_steps / 2)
batch_size = 500  # items sampled from memory to train
enable_double_dqn = False

logger = logging.getLogger(__name__)

# Define the DQN Agent in PyTorch
class Player:

    def __init__(self, name='DQN', load_model=None, env=None, batch_size=64, gamma=0.99, lr=0.001, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, memory_limit=2000):
        self.name = name
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_limit)

        # Load model if specified
        if load_model:
            self.load(load_model)

        # Optimizer
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def mount(self, env):
        self.state_size = env.observation_space[0]
        self.action_size = env.action_space.n

        # Define policy and target networks
        self.policy_net = DQNetwork(self.state_size, self.action_size)
        self.target_net = DQNetwork(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to inference mode

    def start_step_policy(self, env, observation):
        """Custom policy for random decisions for warm up."""
        logger.info("Random action")
        _ = observation
        action = env.action_space.sample()
        return action
    
    def test(self, nb_episodes=5):
        """Evaluate the agent"""
        for episode in range(nb_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.action(self.env.legal_moves, state, None)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward

            print(f"Test Episode {episode+1}/{nb_episodes} - Reward: {episode_reward:.2f}")

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        if np.random.rand() <= self.epsilon:
            # Explore by choosing a random valid action
            return random.choice(action_space)
        
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        allowed_actions = list(this_player_action_space.intersection(set(action_space)))
        if Stage.SHOWDOWN in allowed_actions:
            allowed_actions.remove(Stage.SHOWDOWN)
        if Stage.SHOWDOWN.value in allowed_actions:
            allowed_actions.remove(Stage.SHOWDOWN.value)
        state = torch.FloatTensor(observation).unsqueeze(0)
        
        # Get Q-values from the policy network for the current state
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze().numpy()
        
        # Filter Q-values to only include valid actions
        # allowed_actions = list(action_space)
        # masked_q_values = {action: q_values[action.value] for action in allowed_actions}
        mask = np.full_like(q_values, -np.inf)
        for action in allowed_actions:
            mask[action.value] = q_values[action.value]
        # Select the action with the highest Q-value among allowed actions
        action = np.argmax(mask)
        
        # action = random.choice(list(possible_moves))
        return action

    def replay(self):
        """Train the agent with a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
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
        current_q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Q values for next states (target net)
        next_q_values = self.target_net(next_states).max(1)[0]
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self, filepath):
        """Load the model weights"""
        self.policy_net.load_state_dict(torch.load(filepath))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        """Save the model weights"""
        torch.save(self.policy_net.state_dict(), filepath)