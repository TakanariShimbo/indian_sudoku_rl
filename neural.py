import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import *


class DQNetwork(nn.Module):
    """Deep Q-Network model"""

    def __init__(self, n_features, n_actions, hidden_size=HIDDEN_SIZE):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, 0, 0.3)
            nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DeepQNetwork:
    """Deep Q-Network agent with experience replay and target network"""

    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=LEARNING_RATE,
        reward_decay=REWARD_DECAY,
        e_greedy=EPSILON_GREEDY,
        replace_target_iter=REPLACE_TARGET_ITER,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        e_greedy_increment=EPSILON_INCREMENT,
    ):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment

        # Initialize epsilon
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max

        # Counters
        self.learn_step_counter = 0
        self.memory_counter = 0

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Networks
        self.eval_net = DQNetwork(n_features, n_actions).to(self.device)
        self.target_net = DQNetwork(n_features, n_actions).to(self.device)

        # Optimizer
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=learning_rate)

        # Loss function
        self.loss_func = nn.MSELoss()

        # Initialize target network
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # Cost history
        self.cost_history = []

    def store_transition(self, state, action, reward, next_state):
        """Store experience in replay memory"""
        transition = (state, action, reward, next_state)
        self.memory.append(transition)
        self.memory_counter += 1

    def choose_action(self, observation):
        """Choose action using epsilon-greedy policy"""
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        if np.random.uniform() < self.epsilon:
            # Greedy action
            with torch.no_grad():
                q_values = self.eval_net(observation)
                action = q_values.argmax().item()
        else:
            # Random action
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        """Train the network using experience replay"""
        # Update target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # Sample batch from memory
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        # numpy array
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])

        # numpy array to tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Current Q values
        current_q_values = self.eval_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = self.loss_func(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store cost
        self.cost_history.append(loss.item())

        # Update epsilon
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)

        self.learn_step_counter += 1

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save(
            {
                "eval_net_state_dict": self.eval_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step_counter": self.learn_step_counter,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.eval_net.load_state_dict(checkpoint["eval_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step_counter = checkpoint["learn_step_counter"]
        print(f"Model loaded from {filepath}")
