from snake_env import Snake_env
import numpy as np
import random
from neuralNetwork import NeuralNetwork, mutatedNeuralNetwork
from copy import deepcopy
import pickle
import time

probs = np.array([0.1, 0.1, 0.8])

def step_agent():
    global score
    global random_apples
    global random_snakes

    action = np.argmax(population[0].forward(env.getInput()))
    state, reward, done, _ = env.step(action)

    if (reward > 1):
        score += 1

    if done:
        env.random_apples = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]
        env.random_snakes = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]

        state = env.reset()

        print(score)

        score = 0

    return state

def play():
    global score

    state2 = step_agent()

    state = state2

population = []

with open("bestAgent.pkl", "rb") as file:
    population.append(pickle.load(file))

width = 20
height = 20

window_size = 800

random_apples = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]
random_snakes = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]

env = Snake_env(width, height, window_size, window_size, 2, random_apples, random_snakes, pygame_env=True)

state = env.reset()
score = 0

while True:
    play()
    env.render(score=round(score, 2))

    #time.sleep(0.1)
