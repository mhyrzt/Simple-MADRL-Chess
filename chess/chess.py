import gym
import pygame
import numpy as np
from typing import Union
from pygame.font import Font
from pygame.surface import Surface

from chess.types import Cell, Action, Trajectory
from chess.pieces import Pieces
from chess.colors import Colors
from chess.rewards import Rewards
from chess.info_keys import InfoKeys


class Chess(gym.Env):
    metadata: dict = {
        "render_mode": ("human", "rgb_array"),
    }

    def __init__(
        self,
        max_steps: int = 150,
        render_mode: str = "human",
        window_size: int = 800,
    ) -> None:
        self.turn: int = Pieces.WHITE
        self.done: bool = False
        self.checked: bool = [False, False]
        self.board: np.ndarray = self.init_board()
        self.steps: int = 0
        self.max_steps: int = max_steps

        self.cell_size: int = window_size // 8
        self.window_size: int = window_size

        self.font: Font = None
        self.screen: Surface = None
        self.render_mode: str = render_mode

        self.init_pygame()

    def init_board(self) -> np.ndarray:
        board = np.zeros((2, 8, 8), dtype=np.uint8)
        board[:, 0, 3] = Pieces.QUEEN
        board[:, 0, 4] = Pieces.KING
        board[:, 1, :] = Pieces.PAWN
        board[:, 0, (0, 7)] = Pieces.ROOK
        board[:, 0, (1, 6)] = Pieces.KNIGHT
        board[:, 0, (2, 5)] = Pieces.BISHOP
        return board

    def draw_cells(self):
        for y in range(8):
            for x in range(8):
                self.draw_cell(x, y)

    def draw_pieces(self):
        for y in range(8):
            for x in range(8):
                self.draw_piece(x, y)

    def render(self) -> Union[None, np.ndarray]:
        self.screen.fill(Colors.BLACK)
        self.draw_cells()
        self.draw_pieces()

        if self.render_mode == "human":
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def init_pygame(self) -> None:
        if self.screen is not None:
            return
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font("chess/seguisym.ttf", self.cell_size // 2)
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size,) * 2)
            pygame.display.set_caption("Chess RL Environment")
        else:
            self.screen = pygame.Surface((self.window_size,) * 2)

    def get_cell_color(self, x: int, y: int) -> tuple[int]:
        if (x + y) % 2 == 0:
            return Colors.GRAY
        return Colors.BLACK

    def get_left_top(self, x: int, y: int, offset: float = 0) -> tuple[int]:
        return self.cell_size * x + offset, self.cell_size * y + offset

    def draw_cell(self, x: int, y: int) -> None:
        pygame.draw.rect(
            self.screen,
            self.get_cell_color(x, y),
            pygame.Rect((*self.get_left_top(x, y), self.cell_size, self.cell_size)),
        )

    def draw_piece(self, x: int, y: int) -> None:
        row, col = y, x
        for color in [Pieces.BLACK, Pieces.WHITE]:

            if self.is_empty((row, col), color):
                continue

            yy = abs((color * 7) - y)
            text = self.font.render(
                Pieces.get_ascii(color, self.board[color, row, col]),
                True,
                Colors.WHITE,
                self.get_cell_color(x, yy),
            )
            rect = text.get_rect()
            rect.center = self.get_left_top(x, yy, offset=self.cell_size // 2)
            self.screen.blit(text, rect)

    def close(self) -> None:
        if self.screen is None:
            return
        pygame.display.quit()
        pygame.quit()

    def reset(self) -> np.ndarray:
        self.turn = Pieces.WHITE
        self.steps = 0
        self.board = self.init_board()
        self.checked = [False, False]
        return self.board

    def check_for_enemy(self, cell: Cell) -> bool:
        row, col = cell
        return not self.is_empty((7 - row, col), 1 - self.turn)

    def check_for_enemy_king(self, cell: Cell) -> bool:
        row, col = cell
        return self.board[1 - self.turn, 7 - row, col] == Pieces.KING

    def empty_enemy_cell(self, cell: Cell) -> None:
        row, col = cell
        self.board[1 - self.turn, 7 - row, col] = Pieces.EMPTY

    def move_piece(self, current_cell: Cell, next_cell: Cell):
        nr, nc = next_cell
        cr, cc = current_cell
        self.board[self.turn, nr, nc] = self.board[self.turn, cr, cc]
        self.board[self.turn, cr, cc] = Pieces.EMPTY

    def promote_pawn(self, cell: Cell):
        row, col = cell
        if self.board[self.turn, row, col] == Pieces.PAWN and row == 7:
            self.board[self.turn, row, col] = Pieces.QUEEN

    def is_wrong_move(
        self, current_cell: Cell, next_cell: Cell, color: int = None
    ) -> bool:
        color = self.turn if (color is None) else color
        row, col = current_cell
        cond_1 = self.is_empty(next_cell, color)
        cond_2 = not self.check_for_enemy_king(next_cell)
        cond_3 = Pieces.validate_move(
            self.board[color, row, col],
            current_cell,
            next_cell,
            self.check_for_enemy(next_cell),
        )
        return not (cond_1 and cond_2 and cond_3)

    def is_empty(self, cell: Cell, color: int):
        row, col = cell
        return self.board[color, row, col] == Pieces.EMPTY

    def get_enemy_king_pos(self) -> Cell:
        where = np.where(self.board[1 - self.turn] == Pieces.KING)
        row, col = np.concatenate(where)
        return row, col

    def is_check_piece(self, enemy_king_pos: Cell, piece_pos: Cell) -> bool:
        rp, cp = piece_pos
        rk, ck = enemy_king_pos
        return Pieces.validate_move(
            self.board[self.turn, rp, cp], piece_pos, (7 - rk, ck), True
        )

    def is_path_empty(self, enemy_king_pos: Cell, piece_pos: Cell) -> bool:
        rp, cp = piece_pos
        piece = self.board[self.turn, rp, cp]

        if piece == Pieces.ROOK:
            return self.is_path_empty_rook(enemy_king_pos, piece_pos)

        if piece == Pieces.BISHOP:
            return self.is_path_empty_bishop(enemy_king_pos, piece_pos)

        if piece == Pieces.QUEEN:
            return self.is_path_empty_queen(enemy_king_pos, piece_pos)

        # PAWN OR KNIGHT
        return True

    def is_both_side_empty(self, pos: Cell, turn: int) -> bool:
        r, c = pos
        c1 = self.is_empty((r, c), turn)
        c2 = self.is_empty((7 - r, c), 1 - turn)
        c3 = self.board[1 - turn, 7 - r, c] == Pieces.KING
        return c1 and c2 or c3

    def is_path_empty_queen(self, enemy_king_pos: Cell, piece_pos: Cell) -> bool:
        rp, cp = piece_pos
        rk, ck = enemy_king_pos
        if 7 - rk == rp or ck == cp:
            return self.is_path_empty_rook(enemy_king_pos, piece_pos)
        return self.is_path_empty_bishop(enemy_king_pos, piece_pos)

    def is_path_empty_rook(self, enemy_king_pos: Cell, piece_pos: Cell) -> bool:
        rp, cp = piece_pos
        rk, ck = enemy_king_pos
        rk = 7 - rk

        if rk == rp:
            d = ck - cp
            s = np.sign(d)
            for i in range(1, abs(d)):
                nc = cp + i * s
                if not self.is_both_side_empty((rk, nc), self.turn):
                    return False

        elif ck == cp:
            d = rk - rp
            s = np.sign(d)
            for i in range(1, abs(d)):
                nr = rp + i * s
                if not self.is_both_side_empty((nr, ck), self.turn):
                    return False

        return True

    def is_path_empty_bishop(self, enemy_king_pos: Cell, piece_pos: Cell) -> bool:
        rp, cp = piece_pos
        rk, ck = enemy_king_pos
        rk = 7 - rk
        dr = rk - rp
        dc = ck - cp
        for i in range(1, abs(dr)):
            nr = rp + i * np.sign(dr)
            nc = cp + i * np.sign(dc)
            if not self.is_both_side_empty((nr, nc), self.turn):
                return False
        return True

    def is_check(self, king_pos: Cell = None) -> bool:
        king_pos = self.get_enemy_king_pos() if (king_pos is None) else king_pos
        for re in range(8):
            for ce in range(8):
                if self.is_check_piece(king_pos, (re, ce)):
                    if self.is_path_empty(king_pos, (re, ce)):
                        return True
        return False

    def get_king_next_possible_pos(self) -> list:
        rk, ck = self.get_enemy_king_pos()
        nxt_ps = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                rn = rk + i
                cn = ck + j
                if (-1 < rn < 8) and (-1 < cn < 8):
                    if self.is_empty((rn, cn), 1 - self.turn):
                        nxt_ps.append((rn, cn))
        return nxt_ps

    def is_check_mate(self):
        for next_king_pos in self.get_king_next_possible_pos():
            if not self.is_check(next_king_pos):
                return False
        return True

    def validate_and_move(
        self, current_cell: Cell, next_cell: Cell
    ) -> tuple[float, dict]:
        infos = [{}, {}]
        rewards = [0, 0]
        row, col = current_cell

        if self.checked[self.turn] and self.board[self.turn, row, col] != Pieces.KING:
            infos[self.turn][InfoKeys.WRONG_MOVE] = True
            rewards[self.turn] = Rewards.WRONG_MOVE
            return rewards, infos

        if self.is_empty(current_cell, self.turn):
            infos[self.turn][InfoKeys.EMPTY_SELECT] = True
            rewards[self.turn] = Rewards.EMPTY_SELECT
            return rewards, infos

        if self.is_wrong_move(current_cell, next_cell):
            infos[self.turn][InfoKeys.WRONG_MOVE] = True
            rewards[self.turn] = Rewards.WRONG_MOVE
            return rewards, infos

        rewards = [Rewards.MOVE, Rewards.MOVE]
        self.checked = [False, False]

        self.empty_enemy_cell(next_cell)
        self.move_piece(current_cell, next_cell)
        self.promote_pawn(next_cell)

        if self.is_check_mate():
            infos[self.turn][InfoKeys.CHECK_MATE_WIN] = True
            rewards[self.turn] = Rewards.CHECK_MATE_LOSE

            infos[1 - self.turn][InfoKeys.CHECK_MATE_LOSE] = True
            rewards[1 - self.turn] = Rewards.CHECK_MATE_LOSE

            self.done = True

        elif self.is_check():
            infos[self.turn][InfoKeys.CHECK_WIN] = True
            rewards[self.turn] = Rewards.CHECK_WIN

            infos[1 - self.turn][InfoKeys.CHECK_LOSE] = True
            rewards[1 - self.turn] = Rewards.CHECK_LOSE

            self.checked[self.turn] = False
            self.checked[1 - self.turn] = True

        self.steps += 1
        self.turn = 1 - self.turn

        return rewards, infos

    def step(self, action: Action) -> Trajectory:
        current_cell, next_cell = action
        rewards, infos = self.validate_and_move(current_cell, next_cell)
        done = (self.steps >= self.max_steps) or (self.done)
        return self.board, rewards, done, infos
