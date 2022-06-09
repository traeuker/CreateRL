import numpy as np
import cv2
from PIL import Image

import gym
from gym import spaces

PLAYER_KEY = 1 
REWARD_KEY = 2

color_code = \
    {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red

class grid(gym.Env):
    """
    Simple Grid environment
    One player, one reward on an (per default) 8x8 board.
    """
    def __init__(self, size=(8,8), vision=0):
        self.spec = gym.envs.registration.EnvSpec('GymEnv-v0')
        self.spec.id = "GridEnv"
        self.vision = vision  # if vision states are (5,5), if not (25,)

        self.size = size
        self.action_space = spaces.Discrete(4)
        if vision:
            self.observation_space = spaces.Box(low=-2, high=2, shape=
                    size, dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=-2, high=2, shape=
                    (size[0]*size[1],), dtype=np.uint8)

        self.step_per_episode = 0
        self.n_steps = 50

        _ = self.reset()

    def get_state(self):
        board = np.zeros(self.size) 
        board[self.r_y][self.r_x] = -1
        board[self.y][self.x] = 1
        
        if self.vision:
            return board
        else:
            return board.reshape((self.size[0]*self.size[1],))
        
    def get_reward(self):
        # Action reward given to the agent
        state = self.get_state()
        reward = (-1)
        done = False
        if (self.r_x,self.r_y) == (self.x,self.y):
                reward = 10
                done = True
                _ = self.reset()

        if self.step_per_episode > self.n_steps:
            _ = self.reset() 
        
        return state, reward, done
    def move(self, x, y):
        
        self.x += x
        self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size[0]-1:
            self.x = self.size[0]-1 

        if self.y < 0:
            self.y = 0
        elif self.y > self.size[1]-1:
            self.y = self.size[1]-1 
   
    def reset(self):
        # reset the agent when an episode begins         
        self.board = np.zeros(self.size)
        
        # player pos
        self.x = np.random.randint(0,self.size[0])
        self.y = np.random.randint(0,self.size[1])

        # reward pos
        self.r_x = np.random.randint(0,self.size[0])
        self.r_y = np.random.randint(0,self.size[1])

        if (self.r_x,self.r_y) == (self.x,self.y):
            _ = self.reset()
        s = self.get_state()
        self.step_per_episode = 0
        return s

    def render(self):
        board = np.zeros(self.size+(3,), dtype=np.uint8)
        board[self.y][self.x] = color_code[PLAYER_KEY]
        board[self.r_y][self.r_x] = color_code[REWARD_KEY]
        
        img = Image.fromarray(board, 'RGB')  
        img = img.resize((3000, 3000), Image.NEAREST)

        cv2.imshow("ENV", np.array(img))
        cv2.waitKey(100) & 0xFF
        cv2.destroyAllWindows()

    def close(self):
        cv2.destroyWindow("ENV") 

    def step(self, action):
        # Agent takes the step, i.e. take action to interact with the environment
        info = {}
        if action == 0: 
            self.move(x=1,y=0) # right
        elif action == 1:
            self.move(x=-1,y=0) # left
        elif action == 2:
            self.move(x=0,y=1) # down
        elif action == 3:
            self.move(x=0,y=-1) # up

        self.step_per_episode += 1
        s, r, d = self.get_reward()            
        return s, r, d, info
    



if __name__ == '__main__':
    env = grid(vision=1)
    env.step(0)
    env.action_space.sample()
    print("end")
    # some test
