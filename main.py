from cartpole import CartPole
from cartpole_agent import CartpoleAgent

from atari_games import AtariGames
from atari_agent import AtariAgent

if __name__ == '__main__':
    record = 0
    total_score = 0
    log_itr = 1
    
    game = AtariGames()
    agent = AtariAgent(game)
    
    while True:
        agent.steps += 1
        state_old = agent.get_state()                
        final_move = agent.get_action(state_old)
        reward, done, score =  game.play_step(final_move)
        state_new = agent.get_state()

        agent.train_single(state_old, final_move, reward, state_new, done)
        agent.save_in_memory(state_old, final_move, reward, state_new, done)
        
        if done:
            agent.n_games += 1
            agent.train_batch()
            game.reset()

            total_score += score
            mean_score = total_score / agent.n_games
            if agent.n_games % log_itr == 0:
                agent.model.save_model(game.model_file_name)
                print('Game#', agent.n_games, 'Current Score', score, 'Record', record, 'Mean Score', mean_score)                  
            if score > record:
                record = score