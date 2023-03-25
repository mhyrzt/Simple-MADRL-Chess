Cell = tuple[int]


class Pieces:
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

    @staticmethod
    def get_ascii(color: int, piece: int):
        return Pieces.ASCIIS[color][piece - 1]

    @staticmethod
    def validate_move_pawn(current_cell: Cell, next_cell: Cell, enemy: bool) -> bool:
        next_row, next_col = next_cell
        curr_row, curr_col = current_cell
        diff_col = abs(next_col - curr_col)
        cond_1 = (curr_row == 1 and next_row == 3) and (diff_col == 0) and (not enemy)
        cond_2 = (next_row == curr_row + 1) and (diff_col == 0) and (not enemy)
        cond_3 = (next_row == curr_row + 1) and (diff_col == 1) and enemy
        return cond_1 or cond_2 or cond_3

    @staticmethod
    def validate_move_bishop(current_cell: Cell, next_cell: Cell, *args) -> bool:
        next_row, next_col = next_cell
        curr_row, curr_col = current_cell
        diff_row = abs(next_row - curr_row)
        diff_col = abs(next_col - curr_col)
        return diff_row == diff_col

    @staticmethod
    def validate_move_knight(current_cell: Cell, next_cell: Cell, *args) -> bool:
        next_row, next_col = next_cell
        curr_row, curr_col = current_cell
        diff_row = abs(next_row - curr_row)
        diff_col = abs(next_col - curr_col)
        cond_1 = diff_row == 2 and diff_col == 1
        cond_2 = diff_row == 1 and diff_col == 2
        return cond_1 or cond_2

    @staticmethod
    def validate_move_rook(current_cell: Cell, next_cell: Cell, *args) -> bool:
        next_row, next_col = next_cell
        curr_row, curr_col = current_cell
        diff_row = abs(next_row - curr_row)
        diff_col = abs(next_col - curr_col)
        cond_1 = diff_row > 0 and diff_col == 0
        cond_2 = diff_col > 0 and diff_row == 0
        return cond_1 or cond_2

    @staticmethod
    def validate_move_queen(current_cell: Cell, next_cell: Cell, *args) -> bool:
        cond_1 = Pieces.validate_move_rook(current_cell, next_cell)
        cond_2 = Pieces.validate_move_bishop(current_cell, next_cell)
        return cond_1 or cond_2

    @staticmethod
    def validate_move_king(current_cell: Cell, next_cell: Cell, *args) -> bool:
        next_row, next_col = next_cell
        curr_row, curr_col = current_cell
        diff_row = abs(next_row - curr_row)
        diff_col = abs(next_col - curr_col)
        cond_1 = diff_row == 1 and diff_col == 0
        cond_2 = diff_row == 0 and diff_col == 1
        cond_3 = diff_row == 1 and diff_col == 1
        return cond_1 or cond_2 or cond_3

    @staticmethod
    def validate_move_empty(*args):
        return False

    @staticmethod
    def validate_move(
        piece: int, current_cell: Cell, next_cell: Cell, enemy: bool
    ) -> bool:
        return (
            Pieces.validate_move_empty,
            Pieces.validate_move_pawn,
            Pieces.validate_move_bishop,
            Pieces.validate_move_knight,
            Pieces.validate_move_rook,
            Pieces.validate_move_queen,
            Pieces.validate_move_king,
        )[piece](current_cell, next_cell, enemy)
