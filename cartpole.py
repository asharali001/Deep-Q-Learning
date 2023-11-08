import gym
import numpy as np

from cartpole_deep_net import CartpoleDeepNet
class CartPole:
    def __init__(self, render_mode='None'):
        self.env_title = 'CartPole-v1'
        self.env=gym.make(self.env_title, render_mode=render_mode)               
        
        self.single_state_shape_len = 1
        self.desired_mean_score = 130

        self.observation_space_shape = self.env.observation_space.shape
        self.no_of_actions = self.env.action_space.n
        self.hidden_layers = [128]
        self.model = CartpoleDeepNet(self.observation_space_shape[0], self.hidden_layers, self.no_of_actions)
        self.model_file_name = 'cartpole_model.pth'
        
        self.reset()

    def reset(self):
        self.score = 0
        self.state = self.env.reset()

    def get_state(self):
        if len(self.state) != self.observation_space_shape[0]:
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

