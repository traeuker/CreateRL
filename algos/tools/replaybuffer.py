import numpy as np
from collections import namedtuple
from scipy import signal

class ReplayBuffer():

    def __init__(self, config):
        self.size = 0
        self.max_size = config['replay_buffer_size']
        self.batch_size = config['batch_size']
        self.data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self.data = self.data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])

    def add_data(self, s, a, ns, r, d):
        self.data.states.append(s)
        self.data.actions.append(a)
        self.data.next_states.append(ns)
        self.data.rewards.append(r)
        self.data.terminal_flags.append(d)

        if self.size > self.max_size:
            self.data.states.pop(0)
            self.data.actions.pop(0)
            self.data.next_states.pop(0)
            self.data.rewards.pop(0)
            self.data.terminal_flags.pop(0)
        else:
            self.size += 1
                
    def get_batch(self):
        b_indices = np.random.choice(self.size - 1, self.batch_size)

        b_states = np.array([self.data.states[i] for i in b_indices])
        b_actions = np.array([self.data.actions[i] for i in b_indices])
        b_next_states = np.array([self.data.next_states[i] for i in b_indices])
        b_rewards = np.array([self.data.rewards[i] for i in b_indices])
        b_terminal_flags = np.array([self.data.terminal_flags[i] for i in b_indices])
        
        return b_states, b_actions, b_next_states, b_rewards, b_terminal_flags


class TrajectoryBuffer(object):
    def __init__(self, config):
        self.buffer_size = config['update_every']
        self.b_s = np.zeros((self.buffer_size,config['state_space']), dtype=np.float32)
        self.b_a = np.zeros((self.buffer_size,1), dtype=np.float32)
        # This is confusing: it should be single digit (1), because you only do 
        # one action rather than the value of every action 
        self.b_adv = np.zeros(self.buffer_size, dtype=np.float32)
        self.b_r = np.zeros(self.buffer_size, dtype=np.float32)
        self.b_val = np.zeros(self.buffer_size, dtype=np.float32)
        self.b_return = np.zeros(self.buffer_size, dtype=np.float32)
        self.b_logp = np.zeros(self.buffer_size, dtype=np.float32)
        # for logp we only consider the log probability of selecting 
        # a specific action in a state 
        self.gamma, self.lam = config['gamma'], config['lambda']
        self.ptr, self.path_start_idx, self.max_size = 0, 0, config['update_every']

    def discount_cumsum(self, x, discount):
        """
        for computing discounted cumulative sums of vectors.
        input vector x:
            [x0, x1, x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,  
            x1 + discount * x2,
            x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def add_data(self, s, a, r, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """

        # buffer has to have room so you can store
        assert self.ptr < self.max_size 
        self.b_s[self.ptr] = s
        self.b_a[self.ptr] = a
        self.b_r[self.ptr] = r
        self.b_val[self.ptr] = val
        # We store the logp because of debugging purposes 
        self.b_logp[self.ptr] = logp 
        self.ptr += 1

    def finish_path(self, val):
        path_slice = slice(self.path_start_idx, self.ptr)
        # We take a slice from our trajectory buffer, which equal to the last 
        # episode played and append the last value estimated from the value-net   
        
        # Adding value to the rewards is only relevant for the discounted 
        # cumulative summation 
        rews = np.append(self.b_r[path_slice], val) 
        vals = np.append(self.b_val[path_slice], val)
        
        # Generalized Advantage Estimate --> Lambda advantage calculation
        # Delta is the advantage estimate at every timestep 
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # Discounted at every time step with the exponential mean discount 
        self.b_adv[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        # Computes rewards-to-go, to be targets for the value function
        self.b_return[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get_trajectory(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    
        self.ptr, self.path_start_idx = 0, 0
        if any(self.b_adv):
            # If there is an advantage --> normalization trick
            adv_mean = np.mean(self.b_adv)
            adv_std = np.std(self.b_adv)
            self.b_adv = (self.b_adv- adv_mean) / adv_std
        return self.b_s, self.b_a, self.b_return, self.b_adv, self.b_logp 
        

