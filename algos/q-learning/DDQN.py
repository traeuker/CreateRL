import numpy as np
import torch
import torch.nn as nn

import gym
import wandb
import time

from DQN import DQN, Q, vision_Q
from tools import ReplayBuffer, tt, get_default_config, network_update

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DDQN(DQN):
    def __init__(self, config):
        super(DDQN, self).__init__(config)
        self.config = config
        
        if config['vision']:
            self.q = vision_Q(self.config).to(device)
            self.q_target = vision_Q(self.config).to(device)
        else: 
            self.q = Q(self.config).to(device)
            self.q_target = Q(self.config).to(device) 
        
        self.q_target.load_state_dict(self.q.state_dict())

        self.loss_function = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=config['lr'])

        self.replay_buffer = ReplayBuffer(config)        

    def update(self, t):
        """ Here is the main difference between DDQN and DQN. """

        b_s, b_a, b_ns, b_r, b_tf = self.replay_buffer.get_batch()
        b_s, b_a, b_ns, b_r, b_tf = tt(b_s), tt(b_a), tt(b_ns), tt(b_r), tt(b_tf) # Casting everything as Tensor 

        """
        In regular DQN we said:
        max_q_next = torch.max(self.q(tt(b_ns)),1) 
        It is a two fold information which is queried from Q: 
        once the maximal Q-value (of course), but also the action which is 
        attributed to this max Q value. 
        In DDQN this opperation is done by two Q-functions.
        """

        with torch.no_grad():
            # Similar to choosing the action during training we query the action according
            # to the highest Q-value of the _policy_ Q-function.
            b_na = torch.argmax(self.q(b_ns), dim=1)

            # With the _target_ Q-function we take the Q-values according to the actions from the 
            # previous step.
            max_q_next = self.q_target(b_ns).gather(1, b_na.long().unsqueeze(1))
            # max_q_next = self.q(b_ns).gather(1, b_na.long().unsqueeze(1)) 
            
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
    
if __name__ == '__main__':
    import sys 
    sys.path.append("env")
    sys.path.append("algos")
    
    start_time = time.time()

    from grid import grid

    env_name = 'GridEnv' # 'CartPole-v0' or 'Acrobot-v1'
    #env = gym.make(env_name)

    env = grid(size =(6,6),vision=0)

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    config = get_default_config()
    environment_details = {
        'env': env.spec.id,
        'WANDB': 1, # Logging on weights and biases

        'state_space': env.observation_space.shape[0],
        'action_space': env.action_space.n,
        'vision': 0, # If state-space consists of pixels 

        'seed': seed,
    }
    config.update(environment_details)

    if config['WANDB']:
        wandb.init(project=env_name, config=config)
    algo = DDQN(config)

    eval_reward = algo.run(env, render=0)
    print("Average performance: %0.1f \n" %(eval_reward))
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

