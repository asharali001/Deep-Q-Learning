import os
import torch
import numpy as np

from atari_games import AtariGames
from progress_plotter import plot_progress

class AtariPlayer:
    def __init__(self):
       self.game = AtariGames(render_mode='human')

    def play_game(self):
        total_score = 0
        no_games = 0
        plot_scores = []
        plot_mean_scores = []
        
        model = self.load_model()    
        while True:
            final_move = [0 for _ in range(self.game.no_of_actions)]            
            state0 = torch.tensor(np.array(self.game.get_state()), dtype=torch.float).unsqueeze(0)
            prediction = model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

            _, done, score = self.game.play_step(final_move)

            if done:
                no_games += 1
                self.game.reset()
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / no_games
                plot_mean_scores.append(mean_score)
                plot_progress(plot_scores, plot_mean_scores)

    def load_model(self):
        model_folder = 'model'
        model_name = self.game.model_file_name
        file_path = os.path.join(model_folder, model_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError("Model file not found. Make sure it exists in the 'model' folder.")
        
        model = self.game.model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

if __name__ == '__main__':
    game_player = AtariPlayer()
    game_player.play_game()