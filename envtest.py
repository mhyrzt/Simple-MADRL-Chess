import pygame
from chess import Chess
from time import sleep
import numpy as np
import random
import sys

sys.setrecursionlimit(100)
env = Chess(window_size=800)
env.render()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # if event.type == pygame.KEYDOWN:
        #     if event.key != pygame.K_SPACE:
        #         continue
        #     else:
    turn = env.turn
    print("White" if turn else "Black")
    src, dst, mask = env.get_all_actions(turn)
    action = random.sample(list(np.where(mask == 1)[0]), 1)[0]
    print(f"Action = {action}", src[action], dst[action])
    rewards, done, infos = env.step(action)
    print(f"Rewards = {rewards}")
    print(f"Infos = {infos}")
    print("-" * 64)
    env.render()
    if done:
        env.reset()
        print("RESET")


env.close()