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


class Chess(gym.Env):
    metadata: dict = {
        "render_mode": ("human", "rgb_arry"),
    }

    def __init__(
        self,
        max_steps: int = 500,
        render_mode: str = "human",
        window_size: int = 800,
    ) -> None:
        self.turn: int = Pieces.WHITE
        self.board: np.ndarray = self.init_board()

        self.steps: int = 0
        self.max_steps: int = max_steps

        self.cell_size: int = window_size // 8
        self.window_size: int = window_size

        self.font: Font = None
        self.screen: Surface = None
        self.render_mode: str = render_mode

        self._init_pygame()

    def init_board(self) -> np.ndarray:
        board = np.zeros((2, 8, 8), dtype=np.uint8)
        board[:, 0, 3] = Pieces.QUEEN
        board[:, 0, 4] = Pieces.KING
        board[:, 1, :] = Pieces.PAWN
        board[:, 0, (0, 7)] = Pieces.ROOK
        board[:, 0, (1, 6)] = Pieces.KNIGHT
        board[:, 0, (2, 5)] = Pieces.BISHOP
        return board

    def _draw_cells(self):
        for y in range(8):
            for x in range(8):
                self._draw_cell(x, y)

    def _draw_pieces(self):
        for y in range(8):
            for x in range(8):
                self._draw_piece(x, y)

    def render(self) -> Union[None, np.ndarray]:
        self.screen.fill(Colors.BLACK)
        self._draw_cells()
        self._draw_pieces()

        if self.render_mode == "human":
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )

    def _init_pygame(self) -> None:
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

    def _get_cell_color(self, x: int, y: int) -> tuple[int]:
        if (x + y) % 2 == 0:
            return Colors.BLACK
        return Colors.GRAY

    def _get_left_top(self, x: int, y: int, offset: float = 0) -> tuple[int]:
        return self.cell_size * x + offset, self.cell_size * y + offset

    def _draw_cell(self, x: int, y: int) -> None:
        pygame.draw.rect(
            self.screen,
            self._get_cell_color(x, y),
            pygame.Rect(
                (
                    *self._get_left_top(x, y),
                    self.cell_size,
                    self.cell_size
                )
            )
        )

    def _draw_piece(self, x: int, y: int) -> None:
        row, col = y, x
        for color in [Pieces.BLACK, Pieces.WHITE]:
            
            if self._is_empty((row, col), color):
                continue

            yy = abs((color * 7) - y)
            text = self.font.render(
                Pieces.get_ascii(color, self.board[color, row, col]),
                True,
                Colors.WHITE,
                self._get_cell_color(x, yy)
            )
            rect = text.get_rect()
            rect.center = self._get_left_top(x, yy, offset=self.cell_size // 2)
            self.screen.blit(text, rect)

    def close(self) -> None:
        if self.screen is None:
            return
        pygame.display.quit()
        pygame.quit()

    def reset(self) -> np.ndarray:
        self.turn = Pieces.WHITE
        self.board = self.init_board()
        return self.board

    def _check_for_enemy(self, cell: Cell) -> bool:
        row, col = cell
        return not self._is_empty((7 - row, col), 1 - self.turn)

    def _empty_enemy_cell(self, cell: Cell) -> None:
        row, col = cell
        self.board[1 - self.turn, 7 - row, col] = Pieces.EMPTY

    def _move_piece(self, current_cell: Cell, next_cell: Cell):
        t = self.turn
        nr, nc = next_cell
        cr, cc = current_cell
        self.board[t, cr, cc], self.board[t, nr, nc] = self.board[t, nr, nc], self.board[t, cr, cc]

    def _promote_pawn(self, cell: Cell):
        row, col = cell
        if self.board[self.turn, row, col] == Pieces.PAWN and row == 7:
            self.board[self.turn, row, col] = Pieces.QUEEN

    
    def _is_wrong_move(self, current_cell: Cell, next_cell: Cell) -> bool:
        row, col = current_cell
        cond_1 = self._is_empty(next_cell, self.turn)
        cond_2 = Pieces.validate_move(
            self.board[self.turn, row, col],
            current_cell,
            next_cell,
            self._check_for_enemy(next_cell)
        )
        return (not (cond_1 and cond_2))

    def _is_empty(self, cell: Cell, color: int):
        row, col = cell
        return self.board[color, row, col] == Pieces.EMPTY

    def validate_and_move(self, current_cell: Cell, next_cell: Cell) -> tuple[float, dict]:

        if self._is_empty(current_cell, self.turn):
            return Rewards.EMPTY_SELECT, {"empty_select": True}

        if self._is_wrong_move(current_cell, next_cell):
            return Rewards.WRONG_MOVE, {"wrong_move": True}

        self._empty_enemy_cell(next_cell)
        self._move_piece(current_cell, next_cell)
        self._promote_pawn(next_cell)
        self.steps += 1
        self.turn = 1 - self.turn

        return Rewards.MOVE, {}

    def step(self, action: Action) -> Trajectory:
        current_cell, next_cell = action
        reward, info = self.validate_and_move(current_cell, next_cell)
        return self.board, reward, self.steps >= self.max_steps, info
