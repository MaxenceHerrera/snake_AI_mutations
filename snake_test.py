from snake_env import Snake_env
import pygame
import time
import keyboard
import random

width = 20
height = 20

random_apples = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]
random_snakes = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]

env = Snake_env(20, 20, 800, 800, 2, random_apples, random_snakes, pygame_env=True)

action = 1

env.reset()

score = 0

while True:
    tm = time.time()

    action = 1

    if keyboard.is_pressed("d"):
        env.direction = (1, 0)

    elif keyboard.is_pressed("w"):
        env.direction = (0, -1)

    elif keyboard.is_pressed("a"):
        env.direction = (-1, 0)

    elif keyboard.is_pressed("s"):
        env.direction = (0, 1)

    _, reward, done, _ = env.step(1)

    if (reward > 1):
        score += 1

    if done:
        print(score)

        score = 0

        env.reset()

    env.render(score=score)

    start = time.time()

    while time.time() - start < 0.1:
        if keyboard.is_pressed("d"):
            env.direction = (1, 0)

        elif keyboard.is_pressed("w"):
            env.direction = (0, -1)

        elif keyboard.is_pressed("a"):
            env.direction = (-1, 0)

        elif keyboard.is_pressed("s"):
            env.direction = (0, 1)