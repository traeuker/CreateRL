import numpy as np
from collections import namedtuple


class ReplayBuffer():

    def __init__(self, config):
        self.size = 0
        self.max_size = config['replay_buffer_size']
        self.batch_size = config['batch_size']
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "terminal_flags"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], terminal_flags=[])

    def add_data(self, s, a, ns, r, d):
        self._data.states.append(s)
        self._data.actions.append(a)
        self._data.next_states.append(ns)
        self._data.rewards.append(r)
        self._data.terminal_flags.append(d)

        if self.size > self.max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)
        else:
            self.size += 1
                
    def random_batch(self):
        b_indices = np.random.choice(self.size - 1, self.batch_size)

        b_states = np.array([self._data.states[i] for i in b_indices])
        b_actions = np.array([self._data.actions[i] for i in b_indices])
        b_next_states = np.array([self._data.next_states[i] for i in b_indices])
        b_rewards = np.array([self._data.rewards[i] for i in b_indices])
        b_terminal_flags = np.array([self._data.terminal_flags[i] for i in b_indices])
        
        return b_states, b_actions, b_next_states, b_rewards, b_terminal_flags
