import gym
import numpy as np

from atari_deep_net import AtariDeepNet
class AtariGames:
    def __init__(self, render_mode=None):
        self.env_title = 'PongNoFrameskip-v4'
        if render_mode is not None:
            self.env=gym.make(self.env_title, render_mode=render_mode)
        else:
            self.env=gym.make(self.env_title)
        self.env = gym.wrappers.AtariPreprocessing(
            self.env, 
            noop_max=30, 
            frame_skip=4, 
            screen_size=84, 
            terminal_on_life_loss=False, 
            grayscale_obs=True, 
            grayscale_newaxis=False, 
            scale_obs=False
        )        
        self.env = gym.wrappers.FrameStack(self.env, 4)

        self.single_state_shape_len = 3
        self.desired_mean_score = 50 #TODO(ASHAR): Fix desired_score for this
        
        
        self.observation_space_shape = self.env.observation_space.shape
        self.no_of_actions = self.env.action_space.n
        self.model = AtariDeepNet(self.env.observation_space, self.no_of_actions)
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
        return current_state

    def play_step(self, action):
        move = action.index(1)
        res = self.env.step(move)
        (state, reward, terminated, truncated , _) = res
        done = truncated or terminated        
        self.state = state
        if not done:
            self.score += reward        
        return reward, done, self.score