import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import wandb

from tools import ReplayBuffer, tt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Q(nn.Module):
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


class DQN():

    def __init__(self, config):
        self.config = config
        
        self.q = Q(self.config)
        self.q_target = copy.deepcopy(self.q)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q.parameters(), config['lr'])

        self.replay_buffer = ReplayBuffer(config)

    def get_action(self, state, epsilon):
        # print(self.q(state[None, :]))
        
        if epsilon > np.random.random():
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = torch.argmax(self.q(tt(state))).numpy().tolist()
            #print(a)
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
                break
        if render:
            env.close()
        print(f"{action_array}")
        wandb.log({'mean_reward_test': total_reward})
        pass

    def update(self):
        b_s, b_a, b_ns, b_r, b_tf = self.replay_buffer.random_batch()

        max_q_next = torch.max(self.q(tt(b_ns)),1) 

        y = tt(b_r) + tt(1 - b_tf) * self.config['gamma'] * max_q_next.values # target

        q = self.q(tt(b_s)).gather(1, tt(b_a).long().unsqueeze(1))

        loss = self.loss_function(y, q)

        wandb.log({'mean_q': torch.mean(q).detach().numpy()}) 
        wandb.log({'loss': loss}) 
        
        loss.backward()

        self.optimizer.zero_grad()
        self.optimizer.step()
 
    
    def run(self, env: None):
        s = torch.rand(1,4)
        a = self.q(s)
    
        """
        For me: Within run should only be the for-loop over timesteps.
        """

        s = env.reset()
        
        for t in range(self.config['training_steps']):
            a = self.get_action(s, self.config['epsilon'])
            ns, r, d, _ = env.step(a) 
            
            # save in RB, TODO: terminal_flag excluding time-out termination

            self.replay_buffer.add_data(s, a, ns, r, d)

            # training on RB data
            if t > self.config['training_start']:
                self.update()
            s = ns 
            # every other timestep, testing
            if t % self.config['eval_every'] == 0:
                self.eval(env)

            if d: 
                env.reset()

        return a


if __name__ == '__main__':
    
    env = gym.make('CartPole-v0')
    
    config= {
        'state_space': env.observation_space.shape[0],
        'action_space': env.action_space.n,
        'l1_size': 32,
        'l2_size': 16,
        'epsilon': 0.1,
        'replay_buffer_size': 25_600,
        'batch_size': 256,
        'gamma': 0.99,
        'lr':3e-4,

        'training_steps': 10_000,
        'training_start': 1_000,
        'eval_every': 1_000

        }

    wandb.init(project='cartpole')
    
    wandb.config = config
    algo = DQN(config)


    q_val = algo.run(env)

    print(q_val)
else:
    print("Package loaded.")