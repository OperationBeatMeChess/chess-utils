import numpy as np

import chess
import chess.pgn

from chess_gym.envs import ChessEnv


# Note, this is part of chess_gym.envs.ChessEnv. Maybe make it static 
def get_piece_configuration(board):
    piece_map = np.zeros(64)

    # for square, piece in zip(board.piece_map().keys(), board.piece_map().values()):
    for square, piece in board.piece_map().items():
        piece_map[square] = piece.piece_type * (piece.color * 2 - 1)
    
    return piece_map.reshape((8,8))


def get_game_data(game):
    """
    Take a game and generate training data where X = board_state and Y = move taken.
    A move is given by [from_square, to_square, promotion, drop]
    """
    
    states = []
    actions = []

    board = game.board()
    for move in game.mainline_moves():
        board_state = get_piece_configuration(board)

        # TODO: make a move_to_encoded function?
        action = ChessEnv._move_to_action(move)

        states.append(board_state)
        actions.append(action)
        board.push(move)
        
    states = np.stack(states)
    actions = np.stack([np.array(a) for a in actions])
    return states, actions


def generate_pgn_data(pgn):
    pgn_states, pgn_actions = [], []

    while (game := chess.pgn.read_game(pgn)) is not None:
        states, actions = get_game_data(game)
        pgn_states.append(states)
        pgn_actions.append(actions)

    return pgn_states, pgn_actions


def test():
    pgn_path = '/home/kage/chess-workspace/PGN-data/lichess_KageLobsta_2022-08-07.pgn'
    pgn = open(pgn_path)
    
    pgn_states, pgn_actions = generate_pgn_data(pgn)

    # print(pgn_actions)
    # print(pgn_states)
    

if __name__ == '__main__':
    test()