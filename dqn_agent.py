import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """Experience replay buffer for storing and sampling experiences"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """Neural network for approximating Q-values"""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    """Deep Q-Network agent for career recommendations"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
                 memory_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = ReplayBuffer(memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Q-networks (policy network and target network)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        logger.info(f"Initialized DQNAgent with state_dim={state_dim}, action_dim={action_dim} on device {self.device}")
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy
        
        Args:
            state (torch.Tensor): Current state (resume embedding)
            
        Returns:
            int: Selected action (career index)
        """
        # Ensure state is a tensor on the correct device
        state = state.to(self.device) if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action (exploration)
            action = random.randrange(self.action_dim)
            logger.debug(f"Taking random action: {action} (epsilon: {self.epsilon:.4f})")
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                q_values = self.q_network(state)
                action = torch.argmax(q_values).item()
                logger.debug(f"Taking greedy action: {action} (epsilon: {self.epsilon:.4f})")
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Performed action
            reward: Received reward
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert to tensors if necessary
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        logger.debug(f"Added experience to replay buffer: action={action}, reward={reward:.2f}")
    
    def train(self):
        """
        Train the agent using experiences from the replay buffer
        
        Returns:
            float: Loss value if training occurred, None otherwise
        """
        # Check if enough samples in replay buffer
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample random batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        if random.random() < 0.01:  # 1% chance of updating target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug("Updated target network")
        
        return loss.item()
    
    def save_model(self, path):
        """
        Save the Q-network model
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the Q-network model
        
        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            logger.warning(f"Model file not found at {path}")
            return False
        
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        
        # Verify state_dim and action_dim match
        if checkpoint['state_dim'] != self.state_dim or checkpoint['action_dim'] != self.action_dim:
            logger.warning("Model dimensions do not match current dimensions!")
        
        logger.info(f"Model loaded from {path} with epsilon={self.epsilon:.4f}")
        return True
