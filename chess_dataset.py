from collections import deque
import chess
import chess.pgn
import numpy as np

from torch.utils.data import Dataset, DataLoader

from adversarial_gym.chess_env import ChessEnv


class ChessDataset(Dataset):
    def __init__(self, pgn_file):
        super().__init__()
        self.pgn_file = open(pgn_file)
        self.data = self.generate_pgn_data()

    def generate_pgn_data(self):
        """ Return a list of tuples with (state, turn,  action, result) """
        data = []

        while (game := chess.pgn.read_game(self.pgn_file)) is not None:
            state_maps, turns, actions, results = self.get_game_data(game)
            game_data = list(zip(state_maps, turns, actions, results))
            data.extend(game_data)

        return data

    def get_game_data(self, game):
        """
        Generate a game's training data, where X = board_state and Y1 = action taken, Y2 = result.
        A move is given by [from_square, to_square, promotion, drop]
        """
        states = []
        actions = []
        results = []
        turns = []

        board = game.board()
        result = game.headers['Result']
        result = self.result_to_number(result)
        for move in game.mainline_moves():
            canon_state = self.get_canonical_state(board, board.turn)
            action = ChessEnv.move_to_action(move)

            states.append(canon_state)
            turns.append(board.turn)
            actions.append(action)
            results.append(result)
            board.push(move)
        return states, turns, actions, results
     
    def get_canonical_state(self, board, turn):
        state = ChessEnv.get_piece_configuration(board)
        state = state if turn else -state
        return state

    def result_to_number(self, result):
        if result == "1-0":
            return 1
        if result == "0-1":
            return -1
        else:
            return 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Maybe useful sometime
# def sequence_game_data(states, actions, seqlen = 20):
#     state_deq = deque([np.zeros(64) for i in seqlen],maxlen=seqlen)
#     action_deq = deque([np.zeros(4672) for i in seqlen],maxlen=seqlen)

#     state_seqs = []
#     action_seqs = []
#     for state, action in zip(states,actions):
#         state_deq.append(state)
#         action_deq.append(action)

#         state_seqs.append(np.array(state_deq))
#         action_seqs.append(np.array(action_deq))

#     return state_seqs, action_seqs