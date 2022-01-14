import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import count
import heapq
import math

import gym
import wandb
import time

from DDQN import DDQN
from DQN import Q, vision_Q
from tools import ReplayBuffer, tt, decay, get_default_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
type_of_experience_replay = "rank-based"  # "vanilla", "rank-based" or "sum-tree"


class rankbased_RB():
    """ Indirect, rank-based prioritization where pi = 1/rank(i), where rank(i) is the rank of transition i 
    when the replay memory is sorted according to |δi|. 
    In the original paper implemented with an array-based binary heap.
    Idea by: https://github.com/alexbooth/DDQN-PER"""

    def __init__(self, config):
        self.size = 0
        self.max_size = config['replay_buffer_size']
        self.batch_size = config['batch_size']
        self.memory = []
        self.tiebreaker = count()

    def add_data(self, s, a, ns, r, d, TD_error):
        heapq.heappush(self.memory, (-TD_error, next(self.tiebreaker), s, a, ns, r, d))
        if self.size > self.max_size:
            self.memory = self.memory[:-1]
        else:
            self.size += 1
        heapq.heapify(self.memory)

    def get_batch(self):
        """ Tell em whats happening here. """
        b_indices = range(self.batch_size)

        batch = heapq.nsmallest(self.batch_size, self.memory)
        b_states = np.array([batch[i][2] for i in b_indices])
        b_actions = np.array([batch[i][3] for i in b_indices])
        b_next_states = np.array([batch[i][4] for i in b_indices])
        b_rewards = np.array([batch[i][5] for i in b_indices])
        b_terminal_flags = np.array([batch[i][6] for i in b_indices])
        return b_states, b_actions, b_next_states, b_rewards, b_terminal_flags
     

class sum_tree_RB():
    """ 
    WIP

    Direct, proportional prioritization where pi = |δi|+ e, where e is a small positive constant 
    that prevents the edge-case of transitions not being revisited once their error is zero. 
    Idea by: https://github.com/takoika/PrioritizedExperienceReplay"""
    
    def __init__(self, config, alpha = 0.5):
        self.max_size = config['replay_buffer_size']
        self.tree_level = np.ceil(np.log(self.max_size + 1, 2)) + 1
        self.tree_size = 2 ** (self.tree_level - 1)
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0
        self.memory_size = config['replay_buffer_size']
        self.batch_size = config['batch_size']
        self.alpha = alpha  # suggested values by paper 0, 0.4, 0.5, 0.6, 0.7, 0.8 

    def add_data(self, s, a, ns, r, d, value):
        index = self.cursor
        self.cursor = (self.cursor+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

        self.data[index] = (s, a, ns, r, d)
        self.val_update(index, value ** self.alpha)
    
        
    def get_batch(self):
        #  if self.size < self.batch_size:
        #     return None, None, None
        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = np.random.random() 
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_size/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating
            
        
        self.priority_update(indices, priorities) # Revert priorities

        weights /= max(weights) # Normalize for stability
        
        return b_states, b_actions, b_next_states, b_rewards, b_terminal_flags


class PER(DDQN):

    def __init__(self, config):
        self.config = config
        
        if config['vision']:
            self.q = vision_Q(self.config).to(device)
            self.q_target = vision_Q(self.config).to(device)
        else: 
            self.q = Q(self.config).to(device)
            self.q_target = Q(self.config).to(device) 

        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config['lr'])

        if type_of_experience_replay== "vanilla":
            self.replay_buffer = ReplayBuffer(config)  
        elif type_of_experience_replay== "rank-based":
            self.replay_buffer = rankbased_RB(config)  
        elif type_of_experience_replay== "vanilla":
            self.replay_buffer = ReplayBuffer(config)  

    def update(self, t):
        """ See DDQN.py for further description - only changed to return loss for replay buffer """
        b_s, b_a, b_ns, b_r, b_tf = self.replay_buffer.get_batch()
        b_s, b_a, b_ns, b_r, b_tf = tt(b_s), tt(b_a), tt(b_ns), tt(b_r), tt(b_tf) 

        with torch.no_grad():
            b_na = torch.argmax(self.q(b_ns), dim=1)
            max_q_next = self.q_target(b_ns).gather(1, b_na.long().unsqueeze(1))
            max_q_next = torch.squeeze(max_q_next, 1)
            y = b_r + (1 - b_tf) * self.config['gamma'] * max_q_next

        q = self.q(b_s).gather(1, b_a.long().unsqueeze(1))
        q = torch.squeeze(q, 1)
        loss = self.loss_function(q, y)

        if self.config['WANDB']:
            wandb.log({'mean_q': q.mean().detach().numpy()}) 
            wandb.log({'mean_target_q': max_q_next.mean().detach().numpy()}) 
            wandb.log({'loss': loss}) 
        
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if (t % self.config['target_net_update_freq']) == 0:
            network_update(self.q_target, self.q, 1)
        return (loss.detach().numpy()).astype(float)

    def run(self, env: None):   
        print('Using {} as Q-Network\n'.format(str(self.q)))
        print('Config: {}\n{}'.format(str(self.config), '#' * 80))

        s = env.reset()
        if self.config['vision']:
            s = transform_visual_input(s)
        eval_reward = []

        timesteps_per_episode = 0
        loss = np.nan
        for t in range(self.config['training_steps']):
            timesteps_per_episode += 1 
            epsilon = decay(t, decay_length=self.config['epsilon_decay_length'])

            if self.config['WANDB']:
                wandb.log({'epsilon': epsilon}) 
            
            if epsilon > np.random.random():
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = torch.argmax(self.q(tt(s))).numpy().tolist()
                    pass

            ns, r, d, _ = env.step(a) 

            if self.config['vision']:
                ns = transform_visual_input(ns)
            
            if t > self.config['training_start']:
                loss = self.update(t)
            self.replay_buffer.add_data(s, a, ns, r, d, loss)

            # testing on every xth timestep
            if t % self.config['eval_every'] == 0:
                rew = self.eval(env, render=1)
                eval_reward.append(rew)

            if d or timesteps_per_episode > self.config['training_episode_length']: 
                timesteps_per_episode = 0 
                s = env.reset()
                if self.config['vision']:
                    s = transform_visual_input(s)
            else:
                s = ns
        _ = self.eval(env, render=1) 
        return np.mean(eval_reward)


if __name__ == '__main__':

    from tools import ReplayBuffer, tt, decay, get_default_config, network_update, transform_visual_input
    from gym_minigrid.wrappers import *
        
    start_time = time.time()

    env_name = 'CartPole-v0'  #'CartPole-v0' or 'MiniGrid-Empty-8x8-v0'
    env = gym.make(env_name)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    config = get_default_config()
    environment_details = {
        'env': env.spec.id,
        'WANDB': 0, # Logging on weights and biases

        'state_space': env.observation_space.shape[0],
        'action_space': env.action_space.n,
        'vision': 0, # If state-space consists of pixels 
        'seed': seed,
    }
    config.update(environment_details)
    if config['vision']:
        env = RGBImgPartialObsWrapper(env) # Get pixel observations
        env = ImgObsWrapper(env)

    if config['WANDB']:
        wandb.init(project=env_name, config=config)
    algo = PER(config)
    eval_reward = algo.run(env)
    print("Average performance: %0.1f \n" %(eval_reward))
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
