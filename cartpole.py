import gym
import numpy as np

class CartPole:
    def __init__(self, render_mode='None'):
        self.env=gym.make('CartPole-v1', render_mode=render_mode)
        self.input_parameters = int(np.prod(self.env.observation_space.shape))
        self.no_of_actions = self.env.action_space.n
        self.hidden_layers = [64]
        self.reset()

    def reset(self):
        self.score = 0
        self.state = self.env.reset()

    def get_state(self):
        if len(self.state) != self.input_parameters:
            return self.state[0]
        return self.state

    def play_step(self, action):
        move = action.index(1)
        (state, reward, terminated, truncated , _) = self.env.step(move)
        done = truncated or terminated
        self.state = state
        if not done:
            self.score += 1        
        return reward, done, self.score

