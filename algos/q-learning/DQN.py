import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import time
import wandb

import sys 
sys.path.append("env")
sys.path.append("algos")

# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
# os.environ["PYTHONWARNINGS"] = 'ignore:resource_tracker:UserWarning'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    """ 
    Convolutional NN - WIP 
    """
    def __init__(self, config):
        super(vision_Q, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, \
            out_channels=config['out_channels_l1'], \
            kernel_size=config['kernel_size_l1'])#, stride=4) 
        self.conv2= nn.Conv2d(in_channels=config['out_channels_l1'], \
            out_channels=config['out_channels_l2'] , \
            kernel_size=config['kernel_size_l2'], stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(8)

        # self.fc = nn.Linear(16*12*12,config['action_space'])
        self.fc1 = nn.Linear(784,64)
        self.fc2 = nn.Linear(64,config['action_space'])

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.conv1(state)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.pool(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DQN():

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

        self.replay_buffer = ReplayBuffer(config)      
    
    def eval(self, env, render=1):
        s = env.reset()
        if self.config['vision']:
            s = transform_visual_input(s)
        total_reward = 0
        trajectory = []
        for _ in range(self.config['eval_episode_length']):
            with torch.no_grad(): 
                a = torch.argmax(self.q(tt(s))).numpy().tolist()
            
            trajectory.append(a)
            s, r, d, _ = env.step(a) 
            if self.config['vision']:
                s = transform_visual_input(s)
            if render:
                env.render()
            total_reward += r
            if d: 
                env.reset()  
                break
        if render:
            # pass
            env.close()
        print(trajectory)
        if self.config['WANDB']:
            wandb.log({'total_reward_during_eval_episode': total_reward})
        return total_reward
      
    def update(self, t):
        self.config['updates_counter'] += 1

        b_s, b_a, b_ns, b_r, b_tf = self.replay_buffer.get_batch()
        b_s, b_a, b_ns, b_r, b_tf = tt(b_s), tt(b_a), tt(b_ns), tt(b_r), tt(b_tf) # casting everything as Tensor 
        
        if self.config['vision']:
            b_s = b_s.reshape(self.config['batch_size'], 1, 8, 8)
            b_ns = b_ns.reshape(self.config['batch_size'], 1, 8, 8)
            # for minigrid 
            # b_s = b_s.reshape(256,1,56,56)
            # b_ns = b_ns.reshape(256,1,56,56)

        with torch.no_grad():
            next_q_values = self.q_target(b_ns)            
            max_q_next_target, _ = torch.max(next_q_values, 1)

            y = b_r + (1 - b_tf) * self.config['gamma'] * max_q_next_target 

        # We take the Q values of state-action pairs. This is done by gathering the Q values 
        # of the states indexed by the actions.
        q = self.q(b_s).gather(1, b_a.long().unsqueeze(1))

        q = torch.squeeze(q, 1)

        loss = self.loss_function(q, y)#.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.config['WANDB']:
            wandb.log({'mean_q': q.mean().detach().numpy()}) 
            wandb.log({'mean_target_q': max_q_next_target.mean().detach().numpy()}) 
            wandb.log({'loss': loss}) 
        
        if (self.config['updates_counter'] % self.config['target_net_update_freq']) == 0:
            network_update(self.q_target, self.q, 1)        
    
    def run(self, env: None, render: 1):   
        print('Using {} as Q-Network\n'.format(str(self.q)))
        print('Config: {}\n{}'.format(str(self.config), '#' * 80))

        s = env.reset()
        if self.config['vision']:
            s = transform_visual_input(s)
        eval_reward = []

        timesteps_per_episode = 0
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
            # if r>0:
            #     print("we got it") 


            if self.config['vision']:
                ns = transform_visual_input(ns)

            self.replay_buffer.add_data(s, a, ns, r, d)

            if t > self.config['training_start'] and (t % self.config['update_every']) == 0:
                self.update(t)

            # testing on every xth timestep
            if t % self.config['eval_every'] == 0:
                rew = self.eval(env, render=render)
                eval_reward.append(rew)

            if d or timesteps_per_episode > self.config['training_episode_length']: 
                
                timesteps_per_episode = 0 

                s = env.reset()
                if self.config['vision']:
                    s = transform_visual_input(s)
            else:
                s = ns
        _ = self.eval(env, render=render) 
        return np.mean(eval_reward)


if __name__ == '__main__':
    
    from tools import ReplayBuffer, tt, decay, get_default_config, network_update, transform_visual_input
    from gym_minigrid.wrappers import *
    

    ### lets seee
    # /opt/homebrew/Caskroom/miniforge/base/lib/python3.9/multiprocessing/resource_tracker.py:216: 
    # UserWarning: resource_tracker: There appear to be 6 leaked semaphore objects to clean up at shutdown
    import multiprocessing
    multiprocessing.Queue(1000)
    ### 
    
    from grid import grid
        
    start_time = time.time()

    env_name = 'GridEnv' # 'CartPole-v0' or 'MiniGrid-Empty-8x8-v0'
    # env_name = 'MiniGrid-Empty-8x8-v0' # 'CartPole-v0' or 'MiniGrid-Empty-8x8-v0'
    # env = gym.make(env_name)

    env = grid(size =(6,6),vision=0)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    config = get_default_config()
    environment_details = {
        'env': env.spec.id,
        'WANDB': 1, # Logging on weights and biases

        # 'state_space': env.observation_space.shape[0],
        # 'state_space': (env.observation_space.sample()['image']).shape,
        'action_space': env.action_space.n,
        'vision': 0, # If state-space consists of pixels 
        
        # these can be defaulted for MiniGird envs 
        'kernel_size_l1': 3,
        'kernel_size_l2': 2,

        'batch_size': 256,

        'seed': seed,
    }
    config.update(environment_details)
    
    if config['vision']:
        # env = RGBImgPartialObsWrapper(env) # Get pixel observations
        # env = ImgObsWrapper(env) 
        try:
            environment_details = {'state_space' : \
                env.observation_space.sample()['image'].shape}

        except IndexError: 
            environment_details = {'state_space' : \
                env.observation_space.sample().shape} 
    else:
       environment_details = {'state_space' : \
            env.observation_space.shape[0]} 

    config.update(environment_details)
    if config['WANDB']:
        wandb.init(project=env_name, config=config)
    algo = DQN(config)
    eval_reward = algo.run(env, render=0)
    print("Average performance: %0.1f \n" %(eval_reward))
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
else:
    from tools import ReplayBuffer, tt, decay, network_update
 