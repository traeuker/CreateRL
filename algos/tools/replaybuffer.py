import numpy as np
from collections import namedtuple

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

