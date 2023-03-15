import pygame
from chess import Chess
from chess.pieces import Pieces
from time import sleep
chess = Chess(window_size=512)
actions = [
    # (Current Cell, Next Cell)
    ((1, 0), (3, 0)), # WHITE
    ((1, 1), (3, 1)), # BLACK
    ((3, 0), (4, 1)), # WHITE
    ((0, 2), (1, 1)), # BLACK
    ((4, 1), (5, 1)), # WHITE
    ((1, 1), (6, 6)), # BLACK
    ((5, 1), (6, 1)), # WHITE
    ((6, 6), (7, 7)), # BLACK
    ((6, 1), (7, 0)), # WHITE
]
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    chess.render()
    turn = "White" if chess.turn == Pieces.WHITE else "Black"
    if len(actions):
        action = actions.pop(0)
        print(f"{turn} Action =", *action)
        print(chess.step(action))
        print("-" * 75)
    sleep(0.5)
pygame.quit()
