 # Reinforcement learning agent (DQN, PPO, etc.)
 
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_rl_agent(env, q_network, episodes=1000):
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = np.argmax(q_values.detach().numpy())  # Choose action with max Q-value
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            next_q_values = q_network(next_state_tensor)
            target_q_value = reward + (0.99 * torch.max(next_q_values)) * (1 - done)
            
            loss = criterion(q_values[0, action], target_q_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state

        print(f"Episode {episode} - Total Reward: {total_reward}")
