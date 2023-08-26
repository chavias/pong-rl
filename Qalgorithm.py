import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from paddle import Paddle
import game

# Define your neural network architecture for the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define your DQN AIPaddle agent using PyTorch
class AIPaddle(Paddle):
    def __init__(self, input_size, output_size, learning_rate, gamma, replay_buffer_size, batch_size):
        super().__init__()
        # network 
        self.q_network = QNetwork(input_size, output_size)
        self.target_q_network = QNetwork(input_size, output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        # optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        # hyperparamter
        self.gamma = gamma        
        # replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        # batch size
        self.batch_size = batch_size


    def choose_action(self, state):
        """ following the gradient without exploration """
        with torch.no_grad():
            q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
            action = q_values.argmax().item()

        return action
        

    def store_experience(self, state, action, reward, next_state, done):
        # Store experience in replay buffer
        experience = (state, action, reward, next_state, done)
        # Add experience to the replay buffer
        self.replay_buffer.append(experience)
        # If the buffer size exceeds the defined limit, remove the oldest experience
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)


    def sample_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough experiences in the replay buffer
        # Randomly sample a batch of experiences
        batch = np.random.sample(self.replay_buffer, self.batch_size)
        # Separate the batch into individual components (states, actions, rewards, etc.)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones


    def train_q_network(self, batch):
        states, actions, rewards, next_states, dones = batch
        # Convert batch components to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        # Compute Q-values for the current states using the Q-network
        q_values = self.q_network(states)
        selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Compute target Q-values using the target Q-network
        target_q_values = rewards + self.gamma * self.target_q_network(next_states).max(1).values * (1 - dones)
        # Calculate the loss (Mean Squared Error)
        loss = self.loss_fn(selected_q_values, target_q_values.detach())
        # Update the Q-network using backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


    def move(self, ball_y):
        action = self.choose_action(game_state)
        super().move(action * 5) 


    def save_model(self, path):
        # Save model weights
        torch.save(self.q_network.state_dict(), path)


    def load_model(self, path):
        # Load model weights
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

# Define hyperparameters
input_size = 4  # Change this based on your state representation
output_size = 2  # Change this based on your action space
learning_rate = 0.001
gamma = 0.99
replay_buffer_size = 1000
batch_size= 10
target_network_update_frequency = 100
num_episodes = 10

# Initialize DQN agent
dqn_agent = AIPaddle(input_size, output_size, learning_rate, gamma, replay_buffer_size, batch_size)

# Training loop
for episode in range(num_episodes):
    state = game.PongGame.reset_game()
    done = False
    total_reward = 0

    while not done:
        action = dqn_agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        dqn_agent.store_experience(state, action, reward, next_state, done)

        batch = dqn_agent.sample_batch()
        dqn_agent.train_q_network(batch)

        state = next_state
        total_reward += reward

    if episode % target_network_update_frequency == 0:
        dqn_agent.update_target_network()

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Save the trained Q-network
dqn_agent.save_model('dqn_model.pth')
