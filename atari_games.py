import cv2
import gym
import ale_py
import numpy as np

# TODO(Ashar): this is messed, still working on it
class AtariGames:
    def __init__(self, render_mode='None'):
        self.env=gym.make('ALE/Pong-v5', render_mode=render_mode)        
        self.input_parameters = 7056
        self.no_of_actions = self.env.action_space.n
        self.hidden_layers = [512, 256, 128]
        self.model_file_name = 'atari_model.pth'
        self.reset()

    def reset(self):
        self.score = 0
        self.state = self.env.reset()

    def get_state(self):
        if len(self.state) == 2:
            current_state = self.state[0]
        else:
            current_state = self.state
        current_state = cv2.cvtColor(current_state, cv2.COLOR_RGB2GRAY)
        current_state = cv2.resize(current_state, (84, 84))
        return current_state.ravel()

    def play_step(self, action):
        move = action.index(1)
        res = self.env.step(move)
        (state, reward, terminated, truncated , _) = res
        done = truncated or terminated        
        self.state = state
        if not done:
            self.score += 1        
        return reward, done, self.score