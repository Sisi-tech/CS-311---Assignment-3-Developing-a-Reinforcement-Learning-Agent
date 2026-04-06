import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# 1. Load Stock Data
data = pd.read_csv("AAPL.csv")  # Replace with your CSV
# data['MA5'] = data['Close'].rolling(5).mean()
# data['MA20'] = data['Close'].rolling(20).mean()
# data = data.dropna().reset_index(drop=True)

import pandas as pd

data = pd.read_csv("AAPL.csv")
print("Number of rows:", len(data))
print(data.head())


# 2. Custom Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3)  # Hold, Buy, Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.cash = 10000
        self.shares = 0

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        return np.array([row['Close'], self.shares, self.cash], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row['Close']
        reward = 0
        # Execute action
        if action == 1 and self.cash >= price:  # Buy
            self.shares += 1
            self.cash -= price
        elif action == 2 and self.shares > 0:  # Sell
            self.shares -= 1
            self.cash += price
        # Reward: change in portfolio value
        portfolio_value = self.cash + self.shares * price
        reward = portfolio_value
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        return self._get_state(), reward, done, {}


# 3. Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


# 4. DQN Network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 5. Training Hyperparameters
EPISODES = 100
LR = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 32
TARGET_UPDATE = 10

# 6. Initialize
env = TradingEnv(data)
state_dim = 3
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
buffer = ReplayBuffer()

# 7. Training Loop
episode_rewards = []

for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy action selection
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Learn from replay buffer
        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions).unsqueeze(1)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(next_states)
            dones_tensor = torch.FloatTensor(dones)

            current_q = policy_net(states_tensor).gather(1, actions_tensor).squeeze()
            next_q = target_net(next_states_tensor).max(1)[0].detach()
            expected_q = rewards_tensor + GAMMA * next_q * (1 - dones_tensor)
            loss = nn.MSELoss()(current_q, expected_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    episode_rewards.append(total_reward)
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    # Update target network
    if ep % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {ep+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")


# 8. Plot Rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Portfolio Value")
plt.title("DQN Trading Agent Performance")
plt.show()