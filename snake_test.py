from snake_env import Snake_env
import pygame
import time
import keyboard

env = Snake_env(20, 20, 800, 800, 2, pygame_env=True)

action = 1

env.reset()

while True:
    tm = time.time()

    action = 1

    if keyboard.is_pressed("d"):
        action = 2

    elif keyboard.is_pressed("w"):
        action = 1

    elif keyboard.is_pressed("a"):
        action = 0

    _, _, done, _ = env.step(action)

    if done:
        env.reset()

    env.render()

    start = time.time()

    while time.time() - start < 0.2:
        pass