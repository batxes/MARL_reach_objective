import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
from models.neural_network import DQN
import os

class Agent:
    def __init__(self, state_size, action_size, agent_id):
        self.id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return action_values.argmax().item()

    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < 64:
            return
        
        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first, then to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        if np.random.rand() < 0.001:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy_net.state_dict(), os.path.join(path, f'agent_{self.id}_policy.pth'))
        torch.save(self.target_net.state_dict(), os.path.join(path, f'agent_{self.id}_target.pth'))

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(os.path.join(path, f'agent_{self.id}_policy.pth')))
        self.target_net.load_state_dict(torch.load(os.path.join(path, f'agent_{self.id}_target.pth')))
