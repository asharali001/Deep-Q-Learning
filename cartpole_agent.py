import torch
import random
import numpy as np
from collections import deque

from cartpole_net_trainer import CartpoleNetTrainer

MAX_MEMORY = 50_000
BATCH_SIZE = 32
LR = 5e-4

class CartpoleAgent:
    def __init__(self, game):
        self.n_games = 0
        self.steps = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)

        self.game = game
        self.model = game.model
        self.trainer = CartpoleNetTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self):                
        return self.game.get_state()

    def save_in_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_batch(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train(self.game.single_state_shape_len, states, actions, rewards, next_states, dones)

    def train_single(self, state, action, reward, next_state, done):
        self.trainer.train(self.game.single_state_shape_len, state, action, reward, next_state, done)

    def get_action(self, state):
        epsilon = np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        final_move = [0 for _ in range(self.game.no_of_actions)]

        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            index = random.randint(0, (self.game.no_of_actions - 1))
            final_move[index] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
