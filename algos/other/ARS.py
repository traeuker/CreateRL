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

from tools import tt, get_default_config
from grid import grid


class policy():
    """ Matrix M is p x n  
    n - size of state-space 
    p - size of action space 
    """
    def __init__(self, config):
        self.config = config
        self.p, self.n = config['state_space'], config['action_space'] 

        # The matrix is p x n, however we need to transpose the matrix for not 
        # losing the action_space dimension 
        self.m = np.zeros((self.n, self.p)) 
        self.mu = np.zeros((1,self.n))
        self.sigma = np.identity(self.n)
        self.j = 0

    def softmax(self, x):
        e = np.exp(x)
        p = e / np.sum(e)
        return p
    
    def get_action(self, s, delta, training=None, direction=None):
        a = None
        if direction is None:
            a = self.m.dot(s)
        elif direction == "pos":
            a = (self.m + self.config['noise'] * delta).dot(s)
        else: # direction == "neg":
            a = (self.m - self.config['noise'] * delta).dot(s)
        if False: # if training is not None:
            # This was previously the matrix processing for actions, because 
            # the algorithm is for continuous action spaces. We could
            # treat the output of the matrix multiplication as probabilities 
            # over possible actions. However, with this we never manage to go to 
            # narrow enough prob distributions and the agent has a very hard 
            # time learning. 
            # If we treat the ouput as quasi-Q values and argmax over the 
            # outputs of the matrix we get way better results.
            a = self.softmax(a)
            a = np.random.choice(range(self.n), p=a)
        else:
            a = np.argmax(a)
        
        return a 
    
    def sample_deltas(self):
        """Sample δ_1, δ_2, ... , δ_N with iid standard normal entries."""
        return [np.random.randn(*self.m.shape) for _ in range(self.config['number_of_directions'])]

    def update(self, deltas, reward_pos_direction, reward_neg_direction):
        rewards = np.append(reward_pos_direction, reward_neg_direction)
        
        # σ_R is the standard deviation of the rewards used in the update step
        sigmas = rewards.std()

        # Sort the directions δ_k by max{r(πj,k,pos), r(πj,k,neg)}, denote 
        # by δ(k) the k-th largest direction, and by π_(j,(k),pos) and 
        # π_(j,(k),neg) the corresponding policies.
        scores = []
        for k in range(self.config['number_of_directions']):
            scores.append(max(reward_neg_direction[k], reward_pos_direction[k]))

        order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
        rollouts = [(reward_pos_direction[k], reward_neg_direction[k], deltas[k]) for k in order]
        
        step = np.zeros(self.m.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        
        if sigmas != 0:
            self.m += self.config['lr'] / (self.config['number_of_directions'] * sigmas) * step

        if self.config['WANDB']:
            wandb.log({'update_step_mean': abs(step.mean())})
            wandb.log({'update_step_mean_std': abs(step.mean())+abs(step.std())})


class ARS():
    def __init__(self, config):
        self.config = config 
        if config['vision']:
            raise NotImplementedError 
        else: 
            self.policy = policy(self.config)
        
        ## Now all for normalzation of states
        self.n = np.zeros(self.config['state_space'])
        self.mean = np.zeros(self.config['state_space'])
        self.mean_diff = np.zeros(self.config['state_space'])
        self.var = np.zeros(self.config['state_space'])
    
    def normalize(self, s):
        # First update the values
        self.n += 1
        pre_mean = self.mean.copy()
        self.mean += (s - self.mean) / self.n
        self.mean_diff += (s - pre_mean) * (s - self.mean)
        self.var = (self.mean_diff / self.n) # .clip(min = 1e-2)

        # Then actually calculate the normalized value
        if self.var.all():
            s = (s - self.mean) / np.sqrt(self.var)
        else:
            s = (s - self.mean)
        return s
    
    def one_episode_run(self, env, delt=0, direction=None, training=True, render=False):
        # If no direction, it should also have not delta 
        s = env.reset() 
        
        total_reward, trajectory = 0, []
        if training:
            len = self.config['training_episode_length']
        else:
            len = self.config['eval_episode_length']
        
        for _ in range(len):
            s = self.normalize(s)
            a = self.policy.get_action(s, delt, training=training, direction=direction)
            
            # To be able to use discrete actions 
            
            trajectory.append(a)
            s, r, d, _ = env.step(a)
            
            if render: 
                env.render()
            total_reward += r
            if d:
                break
        if render: 
            env.close()
        
        if not training:
            print(trajectory)
            if self.config['WANDB']:
                wandb.log({'total_reward_during_eval_episode': total_reward})
        return total_reward

    def run(self, env: None, render: 1):   
        print('Using a {} matrix as Policy\n'.format(str(self.policy.m.shape)))
        print('Config: {}\n{}'.format(str(self.config), '#' * 80))
        eval_reward = []

        for t in range(self.config['training_steps']):
            reward_pos_direction = np.zeros(self.config['number_of_directions'])
            reward_neg_direction = np.zeros(self.config['number_of_directions'])
            
            # Sample δ1, δ2,..., δN in R^(p×n) with iid standard normal entries.
            deltas = self.policy.sample_deltas() 
            
            # Collect 2N rollouts of horizon H and their corresponding rewards 
            # using the 2N policies
            for k in range(self.config['number_of_directions']):
                reward_pos_direction[k] = \
                    self.one_episode_run(env, deltas[k], 'pos')

            for k in range(self.config['number_of_directions']):
                reward_neg_direction[k] = \
                    self.one_episode_run(env, deltas[k], 'neg')

            self.policy.update(deltas, reward_pos_direction, reward_neg_direction)             
            rew = self.one_episode_run(env, training=None, render=render) 

            eval_reward.append(rew) 
        
        # Just a victory lap at the end of the learning 
        _ = self.one_episode_run(env, training=None, render=1)
        return eval_reward


if __name__ == '__main__':
    start_time = time.time()

    env_name = 'CartPole-v0' # 'CartPole-v0'
    
    if env_name == 'GridEnv': # Environment too complex
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
        'agent': 'Random_Search',
        'seed': seed,
        
        'action_space': env.action_space.n, 
        'vision': 0,  
        
        'lr': 3e-4, # in the paper referred to as alpha 
        
        # Specific to ARS
        'number_of_directions': 5, # Sampled per iteration N
        'noise':3e-2,

        'training_steps': 300,
        'eval_episode_length': 500,        
    }
    config.update(environment_details)
    
    if config['vision']:
        raise NotImplementedError
        # try:
        #     environment_details = {'state_space' : \
        #         env.observation_space.sample()['image'].shape}

        # except IndexError: 
        #     environment_details = {'state_space' : \
        #         env.observation_space.sample().shape} 
    else:
       environment_details = {'state_space' : \
            env.observation_space.shape[0]} 

    config.update(environment_details)
    
    if config['WANDB']:
        wandb.init(project=env_name+"_"+config['agent'], config=config)
    
    algo = ARS(config)
    eval_reward = algo.run(env, render=0)

    print("Average performance: %0.1f \n" %np.mean(eval_reward))
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

 