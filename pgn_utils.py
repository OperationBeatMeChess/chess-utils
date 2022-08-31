
import time
from typing import List, Tuple
import numpy as np
import os 
import chess
import chess.pgn

import pickle
from tqdm import tqdm 

from chess_gym.envs import ChessEnv

def generate_pgn_data(pgn):
    """ Return a list of lists of (state, action) pairs for games"""
    pgn_states, pgn_actions = [], []

    while (game := chess.pgn.read_game(pgn)) is not None:
        state_maps, actions = get_game_data(game)
        pgn_states.append(state_maps)
        pgn_actions.append(actions)

    return pgn_states, pgn_actions

def get_game_data(game):
    """
    Take a game and generate training data where X = board_state and Y = move taken.
    A move is given by [from_square, to_square, promotion, drop]
    """
    
    states = []
    actions = []

    board = game.board()
    for move in game.mainline_moves():
        board_map = board.piece_map()

        action = ChessEnv.move_to_action(move)

        states.append(board_map)
        actions.append(action)
        board.push(move)

    # if states and actions:
    #     # states = np.stack(states) if states else []
    #     # actions = np.stack([np.array(a) for a in actions])
    return states, actions

def pickle_dataset(pgn_dir, save_dir):
    """ Pickle game data (states,actions) for each pgn in directory"""
    
    dataset_name = os.path.split(pgn_dir)[1]
    pkl_path = os.path.join(save_dir, f"{dataset_name}_data.pkl")
    counter = 0
    start_time = time.time()
    with open(pkl_path, 'ab+') as f:
        for pgnfile in tqdm(os.scandir(pgn_dir)):
            if '.pgn' not in pgnfile.name: continue
            pgn = open(pgnfile.path)
          
            while (game := chess.pgn.read_game(pgn)) is not None:
                game_data = get_game_data(game) # states, actions
                pickle.dump(game_data, f)
                counter += 1
    time_elapsed = time.time()-start_time
    print(f"Pickled {counter} games - {time_elapsed} seconds")

if __name__ == '__main__':
    pickle_dataset()
