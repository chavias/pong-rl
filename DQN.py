 
import enviroment
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import cProfile

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = enviroment.GameEngine()

#######################################################################
#                        Replay memory
#######################################################################

Transition = namedtuple('Transition',
                        ('state', 'action_left','action_right', 'next_state', 'reward_left','reward_right'))

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#######################################################################
#                        Agents 
#######################################################################

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 21)
        self.layer2 = nn.Linear(21, 21)
        self.layer3 = nn.Linear(21, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self,n_observations, n_actions):
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                  lr=LR, amsgrad=True)


#######################################################################
#                        Initialization 
#######################################################################

BATCH_SIZE = 12
GAMMA = 0.9#0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 17000
TAU = 0.005   # update rate of target network
LR = 0.01#1e-4

agent_left = Agent(n_observations=6,n_actions=3)
agent_right = Agent(n_observations=6,n_actions=3)

memory = ReplayMemory(10000) 
steps_done = 0

def select_action(agent, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return agent.policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.sample()]], device=device, dtype=torch.long)


rewards_ = []


def plot_rewards(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(rewards_, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#######################################################################
#                        Optimization 
#######################################################################

def optimize_model(agent,agent_id):
    ''' Change this funciton so both agents use the same memory but consider a different reward'''
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    if agent_id == "left":
        action_batch = torch.cat(batch.action_left)
        reward_batch = torch.cat(batch.reward_left)
    elif agent_id == "right":
        action_batch = torch.cat(batch.action_right)
        reward_batch = torch.cat(batch.reward_right)
    else:
        raise ValueError('Invalide agent_id')

    state_action_values = agent.policy_net(state_batch).gather(1, action_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    agent.optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(agent.policy_net.parameters(), 100)
    agent.optimizer.step()

if torch.cuda.is_available():
    num_episodes = 10000
else:
    num_episodes = 30000


def RUN():

    reward_left_sum = 0
    reward_right_sum = 0
    reward_left_sum_list = []
    reward_right_sum_list = []

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action_left = select_action(agent_left,state)
            action_right = select_action(agent_right,state)

            observation, reward_left, reward_right, terminated= env.step(action_left.item(),action_right.item())
            reward_left = torch.tensor([reward_left], device=device)
            reward_right = torch.tensor([reward_right], device=device)

            if reward_left != 0 or reward_right !=0:
                print(f"reward left : {reward_left} , reward right: {reward_right}")
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action_left, action_right, next_state, reward_left, reward_right)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(agent_left,agent_id="left")
            optimize_model(agent_right,agent_id='right')

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict_left = agent_left.target_net.state_dict()
            policy_net_state_dict_left = agent_left.policy_net.state_dict()
            for key in policy_net_state_dict_left:
                target_net_state_dict_left[key] = policy_net_state_dict_left[key]*TAU + target_net_state_dict_left[key]*(1-TAU)
            agent_left.target_net.load_state_dict(target_net_state_dict_left)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict_right = agent_right.target_net.state_dict()
            policy_net_state_dict_right = agent_right.policy_net.state_dict()
            for key in policy_net_state_dict_right:
                target_net_state_dict_right[key] = policy_net_state_dict_right[key]*TAU + target_net_state_dict_right[key]*(1-TAU)
            agent_right.target_net.load_state_dict(target_net_state_dict_right)

            if terminated:
                rewards_.append(reward_left)
                plot_rewards()
                break


    torch.save(agent_left.policy_net.state_dict(), "left.pth")
    torch.save(agent_right.policy_net.state_dict(), "right.pth")
    print('Complete')

    plot_rewards(reward_right_sum_list)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    RUN()


    # import pstats
    # cProfile.run("RUN()",filename='RUN.prof')
    # profiler = pstats.Stats('RUN.prof')
    # profiler.sort_stats('cumulative')
    # profiler.print_stats(5)