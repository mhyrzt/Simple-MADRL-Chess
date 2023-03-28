from chess.types import Cell

EMPTY = 0
PAWN = 1
BISHOP = 2
KNIGHT = 3
ROOK = 4
QUEEN = 5
KING = 6

BLACK = 0
WHITE = 1

ASCIIS = (
    ("♙", "♗", "♘", "♖", "♕", "♔"),
    ("♟︎", "♝", "♞", "♜", "♛", "♚"),
)

def get_ascii(color: int, piece: int):
    return ASCIIS[color][piece - 1][0]
