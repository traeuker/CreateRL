import sys
import numpy as np   
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import time
import wandb

sys.path.append("env")
sys.path.append("algos")

from tools import tt, get_default_config, TrajectoryBuffer
from gym_minigrid.wrappers import *

from grid import grid

# For issue: 
# UserWarning: resource_tracker: There appear to be 6 leaked semaphore objects to clean up at shutdown
import multiprocessing
multiprocessing.Queue(1000)
### 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Policy_Net(nn.Module):
    """ Fully connected NN """
    def __init__(self, config):
        super(Policy_Net, self).__init__()

        self.fc1 = nn.Linear(config['state_space'], config['l1_size'])
        self.fc2 = nn.Linear(config['l1_size'], config['l2_size'])
        self.fc3 = nn.Linear(config['l2_size'], config['action_space'])
    
        self.act = torch.tanh 

    def forward(self, x, prev_a=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        
        # The output of your NN (x) is the actions un-normalized log 
        # probabilities.
        a_dist = torch.distributions.Categorical(logits=x)
        a = a_dist.sample()
        
        # The log_prob_a is the logarithm of the probability of an action 
        # given our current policy distribution. We use this for the estimate 
        # of the policy gradient.  
        if prev_a is not None:
            log_prob_a = a_dist.log_prob(prev_a)
        else: 
            log_prob_a = a_dist.log_prob(a)

        # We only return the action distribution for debugging purposes.
        return a, a_dist, log_prob_a


class Value_Net(nn.Module):
    def __init__(self, config):
        super(Value_Net, self).__init__()
        self.fc1 = nn.Linear(config['state_space'], config['l1_size'])
        self.fc2 = nn.Linear(config['l1_size'], config['l2_size'])
        self.fc3 = nn.Linear(config['l2_size'], 1)

        self.act = torch.tanh 

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class vanilla_policy_gradient():
    def __init__(self, config):
        self.config = config 
        if config['vision']:
            raise NotImplementedError 
        else: 
            self.policy_net = Policy_Net(self.config).to(device) 
            self.value_net = Value_Net(self.config).to(device)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config['lr'])
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config['lr_value_function'])

        self.replay_buffer = TrajectoryBuffer(self.config)
        
    def calculate_policy_loss(self, s, a, adv, logp_old):
        _, action_dist, action_log_prob = self.policy_net(s, a) 
        loss = - (action_log_prob * adv).mean()

        if self.config['WANDB']:
            # Useful extra info
            # SANITY CHECK: both should go down over time?
            # Entropy reduces as single action become more probable in some state
            # --> "Low surprise what action chosen"
            # KL reduces as probabilities become more similar over time
            approx_kl = (logp_old - action_log_prob).mean().item()
            ent = action_dist.entropy().mean().item()
            wandb.log({'approx_kl': approx_kl}) 
            wandb.log({'entropy': ent})
        return loss 

    def calculate_value_loss(self, s, ret):
        loss = ((self.value_net(s) - ret)**2).mean()
        return loss

    def update(self, t):
        # Get trajectories 
        b_s, b_a, b_return, b_adv, b_logp = self.replay_buffer.get_trajectory()
        b_s, b_a, b_return, b_adv, b_logp = \
            tt(b_s), tt(b_a), tt(b_return), tt(b_adv), tt(b_logp)
        
        if self.config['WANDB']:
            # Get the loss before update
            with torch.no_grad():
                value_loss_before_update = ((self.value_net(b_s) - b_return)**2).mean()
        
        self.policy_optimizer.zero_grad()
        _, action_dist, action_log_prob = self.policy_net(b_s, b_a) 
        # We take the log prob of the action and look 
        policy_loss = - (action_log_prob * b_adv).mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update value parameter
        for _ in range(self.config['value_function_learning_repetition']):
            self.value_optimizer.zero_grad()
            value_loss = self.calculate_value_loss(b_s, b_return) 
            value_loss.backward()
            self.value_optimizer.step()
        
        if self.config['WANDB']:
            wandb.log({'value_max': max(self.value_net(b_s))}) 
            wandb.log({'value_avg': np.mean(self.value_net(b_s).detach().numpy())}) 

            # Useful extra info
            wandb.log({'policy_loss': policy_loss.item()})  
            wandb.log({'value_loss': value_loss.item()})  
            wandb.log({'delta_value_loss': value_loss.item() - value_loss_before_update})  
            
            # SANITY CHECK: both should go down over time
            # Entropy reduces as single action become more probable in some state
            # --> "Low surprise what action chosen"
            # KL reduces as probabilities become more similar over time
            approx_kl = (b_logp - action_log_prob).mean().item()
            ent = action_dist.entropy().mean().item()
            wandb.log({'approx_kl': approx_kl}) 
            wandb.log({'entropy': ent})


    def eval(self, env, render=1):
        s = env.reset()
        
        total_reward = 0
        trajectory = []
        for _ in range(self.config['eval_episode_length']):
            with torch.no_grad():
                a, _, _ = self.policy_net(tt(s)) 
            a = int(a)
            trajectory.append(a)
            s, r, d, _ = env.step(a) 
            
            if render:
                env.render()
            total_reward += r
            
            if d: 
                env.reset()  
                break
        if render:
            env.close()

        print(trajectory)
        if self.config['WANDB']:
            wandb.log({'total_reward_during_eval_episode': total_reward})
        return total_reward
    

    def run(self, env: None, render: 1):   
        print('Using {} as Policy-Network\n'.format(str(self.policy_net)))
        print('Using {} as Value-Network\n'.format(str(self.value_net)))
        print('Config: {}\n{}'.format(str(self.config), '#' * 80))

        s = env.reset() 
        eval_reward, total_reward = [], 0
        timesteps_per_episode, steps_per_update_cycle = 0, 0
        update_counter = 1
        for t in range(self.config['training_steps']):
            timesteps_per_episode += 1
            steps_per_update_cycle += 1  
                       
            with torch.no_grad():
                a, _, a_log_prob = self.policy_net(tt(s))   
                val = self.value_net(tt(s))
            
            a = int(a)
            ns, r, d, _ = env.step(a)
            total_reward += r
            
            timeout = timesteps_per_episode == self.config['training_episode_length']-1
            terminal = d or timeout
            update_cyle_ended = t == self.config['update_every']-1

            if terminal or update_cyle_ended: 
                if timeout or update_cyle_ended:
                    # If the environment was not solved in the last step before 
                    # the update.
                    # If ignored agent becomes more optimistic (avg_value 
                    # estimate at max) but performs worse (eval_avg at 1/3*max)
                    with torch.no_grad():
                       val = self.value_net(tt(s)).numpy()
                else:
                    val = 0
                self.replay_buffer.finish_path(val)  
                total_reward, timesteps_per_episode = 0, 0  
                s = env.reset()   
                continue
            else:
                s = ns
            
            if t > self.config['update_every'] * update_counter:
                steps_per_update_cycle = 0
                self.update(t)
                update_counter += 1
                
                ## Evaluate at every update
                rew = self.eval(env, render=render) 
                eval_reward.append(rew)  
            self.replay_buffer.add_data(s, a, r, float(val), a_log_prob)
        rew = self.eval(env, render=render) 
        eval_reward.append(rew) 
        return eval_reward


if __name__ == '__main__':
    # code inspired by 
    # - https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg
    start_time = time.time()

    env_name = 'GridEnv' # 'CartPole-v0', 'MiniGrid-Empty-8x8-v0' or 'GridEnv'
    if env_name == 'GridEnv':
        env = grid(size=(5,5),vision=0)
    else:
        env = gym.make(env_name)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    config = get_default_config()
    environment_details = {
        'env': env.spec.id,
        'WANDB': 0, # Logging on weights and biases
        'agent': 'Vanilla_PG',

        'action_space': (env.action_space.n),
        'vision': 0,  
        
        # Mostly for GridEnv in vision-mode 
        # 'kernel_size_l1': 3,
        # 'kernel_size_l2': 2,
        'lr': 3e-4,
        
        'training_steps': 100*4000,
        'eval_episode_length': 1000,
        'update_every': 4000,
        'eval_every': 4000,
        'seed': seed,
    }
    config.update(environment_details)
    
    if config['vision']:
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
        wandb.init(project=env_name+"_"+config['agent'], config=config)
    
    algo = vanilla_policy_gradient(config)
    eval_reward = algo.run(env, render=0)

    print("Average performance: %0.1f \n" %np.mean(eval_reward))
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

 