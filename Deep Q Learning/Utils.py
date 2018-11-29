import gym
from gym.wrappers import Monitor
import gym_ple
from skimage import color, transform

from collections import deque, namedtuple
import random
import torch


class Environment(object):
    def __init__(self, game='FlappyBird-v0', record=False, width=84, height=84, seed=0):
        self.game = gym.make(game)
        self.game.seed(seed)

        if record:
            self.game = Monitor(self.game, './video', force=True)

        self.width = width
        self.height = height

    def play_sample(self, mode: str = 'human'):
        observation = self.game.reset()

        while True:
            screen = self.game.render(mode=mode)
            if mode == 'rgb_array':
                screen = self.preprocess(screen)
            action = self.game.action_space.sample()
            observation, reward, done, info = self.game.step(action)
            if done:
                break
        self.game.close()

    def preprocess(self, screen):
#         preprocessed = screen[:400, 40:]
        preprocessed = screen
        preprocessed = transform.resize(preprocessed, (self.height, self.width))
        preprocessed = color.rgb2gray(preprocessed)
        preprocessed = preprocessed.astype('float32') / 255.

        return preprocessed

    def init(self):
        return self.game.reset()

    def get_screen(self):
        screen = self.game.render('rgb_array')
        screen = self.preprocess(screen)
        return screen

    def step(self, action: int):
        observation, reward, done, info = self.game.step(action)
        return observation, reward, done, info

    def reset(self):
        observation = self.game.reset()
        observation = self.preprocess(observation)
        return observation
    
    def close(self):
        self.game.close()

    @property
    def action_space(self):
        return self.game.action_space.n
    
    @property
    def observation_space(self):
        return self.game.observation_space
    
    
class Memory(object):
    def __init__(self, size, batch_size):
        self.buffer = deque(maxlen=size)
        self.batch_size = batch_size
        self.available = False
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        
    def add(self, state, action, reward, next_state):
        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])
        if next_state is not None:
            next_state = torch.FloatTensor(next_state)
        transition = self.Transition(state=state, action=action, reward=reward, next_state=next_state)
        self.buffer.append(transition)
        
    def sample(self):
        transitions = random.sample(self.buffer, self.batch_size)
        return self.Transition(*(zip(*transitions)))

    def is_ready(self):
        if self.available:
            return True
        else:
            self.available = (len(self.buffer) > self.batch_size)
            return self.available