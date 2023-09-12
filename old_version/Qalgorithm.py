import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from old_version.paddle import Paddle, Hardcode_Paddle
import old_version.game as game
import os.path


# Neural network architecture for the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32) # hiddenlayer
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN AIPaddle agent using PyTorch
class AIPaddle(Paddle):
    def __init__(self,epsilon, input_size, output_size, learning_rate,
                 gamma, replay_buffer_size, batch_size,
                 target_network_update_frequency, num_episodes):
        super().__init__()
        # network
        self.q_network = QNetwork(input_size, output_size)
        self.target_q_network = QNetwork(input_size, output_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        # optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        # hyperparamter
        self.gamma = gamma
        # replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = replay_buffer_size
        # batch size
        self.epsilon = epsilon 
        self.batch_size = batch_size
        self.input_size = input_size  # Change this based on your state representation
        self.output_size = output_size  # Change this based on your action space
        self.learning_rate = learning_rate
        self.replay_buffer_size = replay_buffer_size
        self.target_network_update_frequency = target_network_update_frequency
        self.num_episodes = num_episodes


    def choose_action(self, state):
        """ choosing an action using epsilon greedy exploration """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.output_size)  # Choose a random action
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(
                    list(state.values()), dtype=torch.float32)
                q_values = self.q_network(state_tensor)
                # Choose the action with the highest Q-value (1 or -1)
            action = torch.argmax(q_values).item()  # Choose action with highest Q-value index
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
        batch_indices = np.random.choice(
            len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]
        return batch

    def train_q_network(self, states, actions, rewards, next_states, dones):
        # Convert batch components to PyTorch tensors
        states_list = [torch.tensor(
            list(state.values()), dtype=torch.float32) for state in states]
        states_tensor = torch.stack(states_list)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_list = [torch.tensor(
            list(next_state.values()), dtype=torch.float32) for next_state in next_states]
        next_states_tensor = torch.stack(next_states_list)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        # Calculate Q-values for the current states
        q_values = self.q_network(states_tensor)
        # Calculate Q-values for the next states
        next_q_values = self.target_q_network(next_states_tensor)
        # Calculate the selected Q-values using actions_tensor
        selected_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) 
        # Calculate the target Q-values for training
        target_q_values = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_values.max(1).values
        # Calculate the loss
        loss = F.mse_loss(selected_q_values, target_q_values.detach())
        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def move(self, action):
        #action = self.choose_action(state)
        if action==0:
            super().move(action-1 * 5)
        if action==1:
            super().move(action * 5)
        if action==2:
            super().move(0)

    def save_model(self, path):
        # Save model weights
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        # Load model weights
        self.q_network.load_state_dict(torch.load(path))
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

    def train(self, game):
        print("================================================")
        print(f"Starting training with epsilon = {round(self.epsilon,2)}")
        print("================================================")
        for episode in range(self.num_episodes):
            state = game.get_game_state()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = game.step(action)
                self.store_experience(state, action, reward, next_state, done)

                batch = self.sample_batch()
                if batch is not None:
                    states, actions, rewards, next_states, dones = zip(*batch)
                    self.train_q_network(
                        states, actions, rewards, next_states, dones)

                state = next_state
                total_reward += reward

            if episode % self.target_network_update_frequency == 0:
                self.update_target_network()

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        # Save the trained Q-network
        self.save_model('dqn2.pth')


if __name__ == "__main__":
    # Initialize DQN agent
    epsilon = 0.9
    input_size = 3
    output_size = 3
    learning_rate = 0.006
    gamma = 0.9
    replay_buffer_size = 1000
    batch_size = 32
    target_network_update_frequency = 4
    num_episodes = 100
    paddle1 = Hardcode_Paddle()
    while epsilon >= 0:
        print(f"Learing rate: {learning_rate}")
        aipaddle = AIPaddle(epsilon,input_size, output_size, learning_rate,
                        gamma, replay_buffer_size, batch_size,
                        target_network_update_frequency,
                        num_episodes)
        if os.path.isfile("dqn2.pth"): # if this is changed change network size
            aipaddle.load_model("dqn2.pth")
        Pgame = game.PongGame(paddle1, aipaddle)
        aipaddle.train(Pgame)
        epsilon-=0.05
        learning_rate = round(learning_rate/1.1,5)
