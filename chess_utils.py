from typing import Tuple
import chess 

def get_square_coords(square: int) -> Tuple[int,int]:
    return chess.square_rank(square), chess.square_file(square)

