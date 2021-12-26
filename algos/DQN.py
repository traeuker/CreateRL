# Thanks to: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import wandb

from tools import ReplayBuffer, tt, decay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WANDB = 0

class Q(nn.Module):
    """ Fully connected NN """
    def __init__(self, config):
        super(Q, self).__init__()

        self.fc1 = nn.Linear(config['state_space'], config['l1_size'])
        self.fc2 = nn.Linear(config['l1_size'], config['l2_size'])
        self.fc3 = nn.Linear(config['l2_size'], config['action_space'])
    
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class vision_Q(nn.Module):
    """ Convolutional NN - WIP """
    def __init__(self, config):
        super(vision_Q, self).__init__()


        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2= nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(4*7*7,10)

    def forward(self, state):
        x = self.conv1(state)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class DQN():

    def __init__(self, config):
        self.config = config
        
        if config['vision']:
            self.q = vision_Q(self.config).to(device)
        else: 
            self.q = Q(self.config).to(device)

        self.q_target = copy.deepcopy(self.q).to(device)
        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config['lr'])

        self.replay_buffer = ReplayBuffer(config)

    def get_action(self, state, epsilon):
        # print(self.q(state[None, :]))
        
        if epsilon > np.random.random():
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = torch.argmax(self.q(tt(state))).numpy().tolist()
        return a
    
    def eval(self, env, render = 0):
        s = env.reset()
        total_reward = 0
        action_array = []
        while True:
            a = self.get_action(s, 0) 
            action_array.append(a)
            s, r, d, _ = env.step(a) 
            if render:
                env.render()
            total_reward += r
            if d:
                env.reset()  
                break
        if render:
            env.close()
        # print(f"{action_array}")
        if WANDB:
            wandb.log({'mean_reward_test': total_reward})
        pass

    def update(self):
        b_s, b_a, b_ns, b_r, b_tf = self.replay_buffer.random_batch()

        max_q_next = torch.max(self.q(tt(b_ns)),1) 

        y = tt(b_r) + tt(1 - b_tf) * self.config['gamma'] * max_q_next.values # target

        q = self.q(tt(b_s)).gather(1, tt(b_a).long().unsqueeze(1))

        q = torch.squeeze(q,1)

        loss = self.loss_function(y, q)

        if WANDB:
            wandb.log({'mean_q': torch.mean(q).detach().numpy()}) 
            wandb.log({'loss': loss}) 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
 
    
    def run(self, env: None):   
        s = env.reset()
        
        for t in range(self.config['training_steps']):
            
            epsilon = decay(t)

            a = self.get_action(s, epsilon)

            ns, r, d, _ = env.step(a) 

            self.replay_buffer.add_data(s, a, ns, r, d)

            if t > self.config['training_start']:
                self.update()

            # testing on every other timestep
            if t % self.config['eval_every'] == 0:
                self.eval(env)

            if d: 
                s = env.reset()
            else:
                s = ns
        return a


if __name__ == '__main__':
    
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    config= {
        'state_space': env.observation_space.shape[0],
        'action_space': env.action_space.n,
        'vision': 0, # When state-space consists of pixels 

        'l1_size': 32,
        'l2_size': 16,
        'epsilon': 0.1,
        'replay_buffer_size': 2_560,
        'batch_size': 256,
        'gamma': 0.99,
        'lr':3e-4,

        'training_steps': 100_000,
        'training_start': 1_000,
        'eval_every': 1_000
        }

    if WANDB:
        wandb.init(project=env_name, config=config)
    # wandb.config = config
    algo = DQN(config)


    q_val = algo.run(env)
else:
    print("Package loaded.")