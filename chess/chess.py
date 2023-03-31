import gym
import copy
import pygame
import numpy as np
import chess.moves as Moves
import chess.pieces as Pieces
import chess.colors as Colors
import chess.rewards as Rewards
import chess.info_keys as InfoKeys

from gym import spaces
from typing import Union
from pygame.font import Font
from pygame.surface import Surface

from chess.types import Cell


class Chess(gym.Env):
    metadata: dict = {
        "render_mode": ("human", "rgb_array"),
    }

    def __init__(
        self,
        max_steps: int = 128,
        render_mode: str = "human",
        window_size: int = 800,
    ) -> None:
        self.action_space = spaces.Discrete(640)
        self.observation_space = spaces.Box(0, 7, (128,), dtype=np.int32)

        self.board: np.ndarray = self.init_board()
        self.pieces: list[dict] = self.init_pieces()
        self.pieces_names: list[str] = self.get_pieces_names()

        self.turn: int = Pieces.WHITE
        self.done: bool = False
        self.steps: int = 0
        self.checked: bool = [False, False]
        self.max_steps: int = max_steps

        self.font: Font = None
        self.cell_size: int = window_size // 8
        self.screen: Surface = None
        self.window_size: int = window_size
        self.render_mode: str = render_mode

    def init_board(self) -> np.ndarray:
        board = np.zeros((2, 8, 8), dtype=np.uint8)
        board[:, 0, 3] = Pieces.QUEEN
        board[:, 0, 4] = Pieces.KING
        board[:, 1, :] = Pieces.PAWN
        board[:, 0, (0, 7)] = Pieces.ROOK
        board[:, 0, (1, 6)] = Pieces.KNIGHT
        board[:, 0, (2, 5)] = Pieces.BISHOP
        return board

    def init_pieces(self):
        pieces = {
            "pawn_1": (1, 0),
            "pawn_2": (1, 1),
            "pawn_3": (1, 2),
            "pawn_4": (1, 3),
            "pawn_5": (1, 4),
            "pawn_6": (1, 5),
            "pawn_7": (1, 6),
            "pawn_8": (1, 7),
            "rook_1": (0, 0),
            "rook_2": (0, 7),
            "knight_1": (0, 1),
            "knight_2": (0, 6),
            "bishop_1": (0, 2),
            "bishop_2": (0, 5),
            "queen": (0, 3),
            "king": (0, 4),
        }

        return [pieces.copy(), pieces.copy()]

    def get_state(self, turn: int) -> np.ndarray:
        arr = self.board.copy()
        if turn == Pieces.WHITE:
            arr[[0, 1]] = arr[[1, 0]]
        return arr.flatten()

    def draw_cells(self):
        for y in range(8):
            for x in range(8):
                self.draw_cell(x, y)

    def draw_pieces(self):
        for y in range(8):
            for x in range(8):
                self.draw_piece(x, y)

    def render(self) -> Union[None, np.ndarray]:
        self.init_pygame()
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
        self.done = False
        self.turn = Pieces.WHITE
        self.steps = 0
        self.board = self.init_board()
        self.pieces = self.init_pieces()
        self.checked = [False, False]

    def get_pieces_names(self) -> set:
        return list(self.pieces[0].keys())

    def is_in_range(self, pos: Cell) -> bool:
        row, col = pos
        return row >= 0 and row <= 7 and col >= 0 and col <= 7

    def get_size(self, name: str):
        return Moves.POSSIBLE_MOVES[name]

    def get_empty_actions(self, name: str):
        size = self.get_size(name)
        possibles = np.zeros((size, 2), dtype=np.int32)
        actions_mask = np.zeros((size), dtype=np.int32)
        return possibles, actions_mask

    def is_path_empty(self, current_pos: Cell, next_pos: Cell, turn: int) -> bool:
        next_row, next_col = next_pos
        current_row, current_col = current_pos

        diff_row = next_row - current_row
        diff_col = next_col - current_col
        sign_row = np.sign(next_row - current_row)
        sign_col = np.sign(next_col - current_col)

        size = max(abs(diff_row), abs(diff_col)) - 1
        rows = np.zeros(size, dtype=np.int32) + next_row
        cols = np.zeros(size, dtype=np.int32) + next_col

        if diff_row:
            rows = np.arange(current_row + sign_row, next_row, sign_row, dtype=np.int32)

        if diff_col:
            cols = np.arange(current_col + sign_col, next_col, sign_col, dtype=np.int32)

        for pos in zip(rows, cols):
            if not self.both_side_empty(tuple(pos), turn):
                return False

        return True

    def piece_can_jump(self, pos: Cell, turn: int) -> bool:
        jumps = {Pieces.KNIGHT, Pieces.KING}
        piece = self.board[turn, pos[0], pos[1]]
        return piece in jumps

    def general_validation(
        self,
        current_pos: Cell,
        next_pos: Cell,
        turn: int,
        deny_enemy_king: bool,
    ) -> bool:
        if not self.is_in_range(next_pos):
            return False

        if not self.is_empty(next_pos, turn):
            return False

        if self.is_enemy_king(next_pos, turn) and (not deny_enemy_king):
            return False

        if (not self.piece_can_jump(current_pos, turn)) and (
            not self.is_path_empty(current_pos, next_pos, turn)
        ):
            return False

        return True

    def is_valid_move(
        self,
        current_pos: Cell,
        next_pos: Cell,
        turn: int,
        deny_enemy_king: bool,
    ) -> bool:
        if not self.general_validation(current_pos, next_pos, turn, deny_enemy_king):
            return False
        if self.is_lead_to_check(current_pos, next_pos, turn):
            return False
        return True

    def is_lead_to_check(self, current_pos: int, next_pos: int, turn: int) -> bool:
        temp = Chess(render_mode="rgb_array")
        temp.board = np.copy(self.board)
        temp.move_piece(current_pos, next_pos, turn)
        return temp.is_check(temp.get_pos_king(turn), turn)

    def get_actions_for_bishop(
        self, pos: Cell, turn: int, deny_enemy_king: bool = False
    ):
        possibles, actions_mask = self.get_empty_actions("bishop")
        if pos is None:
            return possibles, actions_mask

        row, col = pos
        for i, (r, c) in enumerate(Moves.BISHOP):
            next_pos = (row + r, col + c)

            if not self.is_valid_move(pos, next_pos, turn, deny_enemy_king):
                continue

            possibles[i] = next_pos
            actions_mask[i] = 1

        return possibles, actions_mask

    def get_actions_for_rook(self, pos: Cell, turn: int, deny_enemy_king: bool = False):
        possibles, actions_mask = self.get_empty_actions("rook")
        if pos is None:
            return possibles, actions_mask

        row, col = pos
        for i, (r, c) in enumerate(Moves.ROOK):
            next_pos = (row + r, col + c)

            if not self.is_valid_move(pos, next_pos, turn, deny_enemy_king):
                continue

            possibles[i] = next_pos
            actions_mask[i] = 1

        return possibles, actions_mask

    def get_action_for_queen(self, pos: Cell, turn: int, deny_enemy_king: bool = False):
        possibles_rook, actions_mask_rook = self.get_actions_for_rook(
            pos, turn, deny_enemy_king
        )
        possibles_bishop, actions_mask_bishop = self.get_actions_for_bishop(
            pos, turn, deny_enemy_king
        )
        possibles = np.concatenate([possibles_bishop, possibles_rook])
        actions_mask = np.concatenate([actions_mask_bishop, actions_mask_rook])

        return possibles, actions_mask

    def get_actions_for_pawn(self, pos: Cell, turn: int, deny_enemy_king: bool = False):
        possibles, actions_mask = self.get_empty_actions("pawn")
        if pos is None:
            return possibles, actions_mask

        row, col = pos
        if self.board[turn, row, col] == Pieces.QUEEN:
            return self.get_action_for_queen(pos, turn)

        for i, (r, c) in enumerate(Moves.PAWN[:4]):
            next_pos = (row + r, col + c)

            if not self.is_valid_move(pos, next_pos, turn, deny_enemy_king):
                continue

            can_moves = (
                (r == 1 and c == 0 and self.both_side_empty(next_pos, turn)),
                (r == 2 and row == 1 and self.both_side_empty(next_pos, turn)),
                (r == 1 and abs(c) == 1 and self.check_for_enemy(next_pos, turn)),
                # TODO: EN PASSANT
            )

            if True in can_moves:
                possibles[i] = next_pos
                actions_mask[i] = 1

        return possibles, actions_mask

    def get_actions_for_knight(
        self, pos: Cell, turn: int, deny_enemy_king: bool = False
    ):
        possibles, actions_mask = self.get_empty_actions("knight")

        if pos is None:
            return possibles, actions_mask

        row, col = pos
        for i, (r, c) in enumerate(Moves.KNIGHT):
            next_pos = (row + r, col + c)
            if not self.is_valid_move(pos, next_pos, turn, deny_enemy_king):
                continue

            possibles[i] = next_pos
            actions_mask[i] = 1

        return possibles, actions_mask

    def get_actions_for_king(self, pos: Cell, turn: int):
        pos
        row, col = pos
        possibles, actions_mask = self.get_empty_actions("king")

        for i, (r, c) in enumerate(Moves.KING):
            next_pos = (row + r, col + c)

            if not self.is_valid_move(pos, next_pos, turn, False):
                continue

            if self.is_neighbor_enemy_king(next_pos, turn):
                continue

            possibles[i] = next_pos
            actions_mask[i] = 1

        return possibles, actions_mask

    def get_source_pos(self, name: str, turn: int):
        cat = name.split("_")[0]
        pos = self.pieces[turn][name]
        if pos is None:
            pos = (0, 0)
        size = self.get_size(cat)
        return np.array([pos] * size)

    def get_actions_for(self, name: str, turn: int, deny_enemy_king: bool = False):
        assert name in self.pieces_names, f"name not in {self.pieces_names}"
        piece_cat = name.split("_")[0]
        piece_pos = self.pieces[turn][name]
        src_poses = self.get_source_pos(name, turn)

        if piece_cat == "pawn":
            return (
                src_poses,
                *self.get_actions_for_pawn(piece_pos, turn, deny_enemy_king),
            )

        if piece_cat == "knight":
            return (
                src_poses,
                *self.get_actions_for_knight(piece_pos, turn, deny_enemy_king),
            )

        if piece_cat == "rook":
            return (
                src_poses,
                *self.get_actions_for_rook(piece_pos, turn, deny_enemy_king),
            )

        if piece_cat == "bishop":
            return (
                src_poses,
                *self.get_actions_for_bishop(piece_pos, turn, deny_enemy_king),
            )

        if piece_cat == "queen":
            return (
                src_poses,
                *self.get_action_for_queen(piece_pos, turn, deny_enemy_king),
            )

        if piece_cat == "king":
            return (
                src_poses,
                *self.get_actions_for_king(piece_pos, turn),
            )

    def get_all_actions(self, turn: int, deny_enemy_king: bool = False):
        all_possibles = []
        all_source_pos = []
        all_actions_mask = []
        for name in self.pieces[turn].keys():
            # DENY ENEMY KING == FOR CHECKMATE VALIDATION ONLY SO ....
            if name == "king" and deny_enemy_king:
                continue

            source_pos, possibles, actions_mask = self.get_actions_for(
                name, turn, deny_enemy_king
            )
            all_source_pos.append(source_pos)
            all_possibles.append(possibles)
            all_actions_mask.append(actions_mask)

        return (
            np.concatenate(all_source_pos),
            np.concatenate(all_possibles),
            np.concatenate(all_actions_mask),
        )

    def check_for_enemy(self, pos: Cell, turn: int) -> bool:
        r, c = pos
        return not self.is_empty((7 - r, c), 1 - turn)

    def is_empty(self, pos: Cell, turn: int) -> bool:
        return self.board[turn, pos[0], pos[1]] == Pieces.EMPTY

    def is_enemy_king(self, pos: Cell, turn: int) -> bool:
        r, c = pos
        return self.board[1 - turn, 7 - r, c] == Pieces.KING

    def both_side_empty(self, pos: Cell, turn: int) -> bool:
        r, c = pos
        return self.is_empty(pos, turn) and self.is_empty((7 - r, c), 1 - turn)

    def get_pos_king(self, turn: int) -> Cell:
        row, col = np.where(self.board[turn] == Pieces.KING)
        return row[0], col[0]

    def is_neighbor_enemy_king(self, pos: Cell, turn: int) -> bool:
        row, col = pos
        row_enemy_king, col_enemy_king = self.get_pos_king(1 - turn)
        row_enemy_king = 7 - row_enemy_king
        diff_row = abs(row - row_enemy_king)
        diff_col = abs(col - col_enemy_king)
        return diff_row <= 1 and diff_col <= 1

    def is_check(self, king_pos: Cell, turn: int) -> bool:
        rk, ck = king_pos
        
        # GO TO UP ROW
        for r in range(rk + 1, 8):
            if not self.is_empty((r, ck), turn):
                break
            p = self.board[1 - turn, 7 - r, ck]
            if p == Pieces.ROOK or p == Pieces.QUEEN:
                return True
        
        # GO TO DOWN ROW
        for r in range(rk - 1, -1, -1):
            if not self.is_empty((r, ck), turn):
                break
            p = self.board[1 - turn, 7 - r, ck]
            if p == Pieces.ROOK or p == Pieces.QUEEN:
                return True
        
        # GO TO RIGHT COL
        for c in range(ck + 1, 8):
            if not self.is_empty((rk, c), turn):
                break
            p = self.board[1 - turn, 7 - rk, c]
            if p == Pieces.ROOK or p == Pieces.QUEEN:
                return True

        # GOT TO LEFT COL
        for c in range(ck - 1, -1, -1):
            if not self.is_empty((rk, c), turn):
                break
            p = self.board[1 - turn, 7 - rk, c]
            if p == Pieces.ROOK or p == Pieces.QUEEN:
                return True

        # CROSS DOWN
        for r in range(rk + 1, 8):
            # RIGHT
            d = r - rk
            for c in [ck + d, ck - d]:
                if not self.is_in_range((r, c)):
                    continue

                if not self.is_empty((r, c), turn):
                    break

                p = self.board[1 - turn, 7 - r, c]
                if p == Pieces.BISHOP or p == Pieces.QUEEN:
                    return True
                
                if d == 1 and p == Pieces.PAWN:
                    return True

        # CROSS UP
        for r in range(rk - 1, -1, -1):
            d = r - rk
            for c in [ck + d, ck - d]:
                if not self.is_in_range((r, c)):
                    continue

                if not self.is_empty((r, c), turn):
                    break

                p = self.board[1 - turn, 7 - r, c]
                if p == Pieces.BISHOP or p == Pieces.QUEEN:
                    return True


        # KNIGHTS
        for r, c in Moves.KNIGHT:
            nr, nc = rk + r, ck + c
            if not self.is_in_range((nr, nc)):
                continue
            if self.board[1 - turn, 7 - nr, nc] == Pieces.KNIGHT:
                return True

        return False

    def update_checks(self, rewards: list[int] = None, infos: list[set] = None):
        rewards = [0, 0] if rewards is None else rewards
        infos = [set(), set()] if infos is None else infos

        for turn in range(2):
            king_pos = self.get_pos_king(turn)
            is_check = self.is_check(king_pos, turn)
            self.checked[turn] = is_check
            if is_check:
                rewards[turn] += Rewards.CHECK_LOSE
                rewards[1 - turn] += Rewards.CHECK_WIN

                infos[turn].add(InfoKeys.CHECK_LOSE)
                infos[1 - turn].add(InfoKeys.CHECK_WIN)
                break
        return rewards, infos

    def update_check_mates(self, rewards: list[int] = None, infos: list[set] = None):
        rewards = [0, 0] if rewards is None else rewards
        infos = [set(), set()] if infos is None else infos

        for turn in range(2):
            _, _, actions = self.get_all_actions(turn)
            if np.sum(actions) == 0:
                self.done = True
                rewards[turn] += Rewards.CHECK_MATE_LOSE
                rewards[1 - turn] += Rewards.CHECK_MATE_WIN

                infos[turn].add(InfoKeys.CHECK_MATE_LOSE)
                infos[1 - turn].add(InfoKeys.CHECK_MATE_WIN)
                break

        return rewards, infos

    def move_piece(self, current_pos: Cell, next_pos: Cell, turn: int):
        next_row, next_col = next_pos
        current_row, current_col = current_pos
        self.board[turn, next_row, next_col] = self.board[
            turn, current_row, current_col
        ]

        self.promote_pawn(next_pos, turn)
        self.board[turn, current_row, current_col] = Pieces.EMPTY
        self.board[1 - turn, 7 - next_row, next_col] = Pieces.EMPTY

        for (key, value) in self.pieces[turn].items():
            if value == tuple(current_pos):
                self.pieces[turn][key] = tuple(next_pos)

        for (key, value) in self.pieces[1 - turn].items():
            if value == (7 - next_pos[0], next_pos[1]):
                self.pieces[1 - turn][key] = None

        rewards = [Rewards.MOVE, Rewards.MOVE]
        rewards[1 - turn] *= 2

        return rewards, [set(), set()]

    def is_game_done(self):
        return self.done or (self.steps >= self.max_steps)

    def promote_pawn(self, pos: Cell, turn: int):
        row, col = pos
        if self.board[turn, row, col] == Pieces.PAWN and row == 7:
            self.board[turn, row, col] = Pieces.QUEEN

    def step(self, action: int):
        assert not self.is_game_done(), "the game is finished reset"
        assert action < 640, "action number must be less than 640"

        source_pos, possibles, actions_mask = self.get_all_actions(self.turn)
        assert actions_mask[action], f"Cannot Take This Action = {action}"
        rewards, infos = self.move_piece(
            source_pos[action], possibles[action], self.turn
        )
        rewards, infos = self.update_checks(rewards, infos)
        rewards, infos = self.update_check_mates(rewards, infos)

        self.turn = 1 - self.turn
        self.steps += 1
        return rewards, self.is_game_done(), infos
