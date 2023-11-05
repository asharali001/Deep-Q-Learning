import gym
import torch
import random
import numpy as np
from collections import deque

from cartpole import CartPole
from q_net_model import QNetModel
from q_net_trainer import QNetTrainer
from progress_plotter import plot_progress

MAX_MEMORY = 50_000
BATCH_SIZE = 32
LR = 5e-4

class Agent:
    def __init__(self, game):
        self.n_games = 0
        self.steps = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.02
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = QNetModel(game.input_parameters, game.hidden_layers, game.no_of_actions)
        self.trainer = QNetTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):                
        return game.get_state()

    def save_in_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_batch(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train(states, actions, rewards, next_states, dones)

    def train_single(self, state, action, reward, next_state, done):
        self.trainer.train(state, action, reward, next_state, done)

    def get_action(self, game, state):
        epsilon = np.interp(self.steps, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])
        final_move = [0 for _ in range(game.no_of_actions)]

        rnd_sample = random.random()
        if rnd_sample <= epsilon:
            index = random.randint(0, (game.no_of_actions - 1))
            final_move[index] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    record = 0
    total_score = 0
    plot_scores = []
    plot_mean_scores = []
    rendering_started = False
    
    game = CartPole()
    agent = Agent(game)
    
    while True:
        agent.steps += 1
        state_old = agent.get_state(game)                
        final_move = agent.get_action(game, state_old)
        reward, done, score =  game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_single(state_old, final_move, reward, state_new, done)
        agent.save_in_memory(state_old, final_move, reward, state_new, done)

        
        if agent.n_games > 0 and (total_score / agent.n_games) > 80:
            # agent.model.save_model()
            if not rendering_started:
                game.env = gym.make('CartPole-v1', render_mode='human')
                game.reset()
                rendering_started = True
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_batch()

            if score > record:
                record = score

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            if agent.n_games % 200 == 0:
                print('Game', agent.n_games, 'Record', record, 'Mean Score', mean_score)                  
                plot_progress(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()