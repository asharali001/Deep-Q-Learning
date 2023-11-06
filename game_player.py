import os
import torch

from cartpole import CartPole
from q_net_model import QNetModel
from progress_plotter import plot_progress

class GamePlayer:
    def __init__(self, game):
       self.game = game

    def play_game(self):
        total_score = 0
        no_games = 0
        plot_scores = []
        plot_mean_scores = []
        
        model = self.load_model()    
        while True:
            final_move = [0 for _ in range(self.game.no_of_actions)]            
            state0 = torch.tensor(self.game.get_state(), dtype=torch.float)
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
        model_name = 'q_net.pth'
        file_path = os.path.join(model_folder, model_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError("Model file not found. Make sure it exists in the 'model' folder.")
        
        model = QNetModel(self.game.input_parameters, self.game.hidden_layers, self.game.no_of_actions)
        model.load_state_dict(torch.load(file_path))
        model.eval()
        return model

if __name__ == '__main__':
    cartpole = CartPole(render_mode='human')
    game_player = GamePlayer(game=cartpole)
    game_player.play_game()