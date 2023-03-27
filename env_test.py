import pygame
from chess import Chess
from chess.pieces import Pieces
from time import sleep

chess = Chess(window_size=512)
actions = [
    ((1, 4), (3, 4)), # WHITE
    ((1, 4), (3, 4)), # BLACK
    ((0, 3), (4, 7)), # WHITE
    ((0, 4), (1, 4)), # BLACK
    ((4, 7), (4, 4)), # WHITE
]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    chess.render()

    if len(actions):
        turn = "White" if chess.turn == Pieces.WHITE else "Black"
        action = actions.pop(0)
        print(turn)
        print(f"Action =", *action)
        print(*chess.step(action)[1:], sep="\n")
        sleep(1)
        print("-" * 70)

chess.close()
