from snake_env import Snake_env
import pygame
import time
import keyboard

env = Snake_env(10, 10, 2)

action = 1

env.reset()

while True:
    tm = time.time()

    if keyboard.is_pressed("s"):
        action = 0

    elif keyboard.is_pressed("d"):
        action = 1

    elif keyboard.is_pressed("w"):
        action = 2

    elif keyboard.is_pressed("a"):
        action = 3

    _, _, done, _ = env.step(action)

    if done:
        action = 1

        env.reset()

    env.render()

    start = time.time()

    while time.time() - start < 0.2:
        if keyboard.is_pressed("s"):
            action = 0

        elif keyboard.is_pressed("d"):
            action = 1

        elif keyboard.is_pressed("w"):
            action = 2

        elif keyboard.is_pressed("a"):
            action = 3