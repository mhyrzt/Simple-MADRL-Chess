import pygame
from chess import Chess
from time import sleep
import numpy as np
import random
import sys

sys.setrecursionlimit(100)
env = Chess(window_size=800)
env.render()
# actions = [
#     ((1, 4), (3, 4)), # WHITE
#     ((1, 4), (3, 4)), # BLACK
#     ((0, 3), (4, 7)), # WHITE
#     ((0, 4), (1, 4)), # BLACK
#     ((4, 7), (4, 4)), # WHITE
# ]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    try:
        turn = env.turn
        _, _, mask = env.get_all_actions(turn)
        action = random.sample(list(np.where(mask == 1)[0]), 1)[0]
        print("White" if turn else "Black")
        print(f"Action = {action}")
        rewards, done, infos = env.step(action)
        print(f"Rewards = {rewards}")
        print(f"Infos = {infos}")
        print("-" * 75)
        env.render()
        if done:
            env.reset()
            print("RESET")
            # sleep(0.25)
    except Exception as e:
        print(e)
        running = False


input("EXIT")
env.close()
