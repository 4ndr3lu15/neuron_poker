import torch
import numpy as np
import random
from gym_env.enums import Action, Stage
import logging
from neural_nets.dqn_nn import DQNetwork

logger = logging.getLogger(__name__)

# Define the DQN Agent in PyTorch
class Player:

    def __init__(self, name='DQN'):
        self.name = name
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995


    def mount(self, env, path=None):
        self.state_size = env.observation_space[0]
        self.action_size = env.action_space.n

        # Define policy and target networks
        self.policy_net = DQNetwork(self.state_size, self.action_size)
        self.target_net = DQNetwork(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to inference mode

        # Load model if specified
        if path:
            self.policy_net.load_state_dict(torch.load(path))

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        if np.random.rand() <= self.epsilon:
            # Explore by choosing a random valid action
            action = random.choice(action_space)
            print("from epsilon")
            print(action)
            
            return action

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
        mask = np.full_like(q_values, -np.inf)
        for action in allowed_actions:
            mask[action.value] = q_values[action.value]
        # Select the action with the highest Q-value among allowed actions
        action = np.argmax(mask)
        action = Action(action)
        print("from policy network")
        print(action)

        return action