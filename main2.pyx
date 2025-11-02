import pyximport; pyximport.install()
from snake_env2 import Snake_env
import numpy as np
import random
from neuralNetwork import NeuralNetwork, mutatedNeuralNetwork
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
import math

def step_agent(i):
    action = np.random.choice(range(3), p=population[i].forward(envs[i].getInput()))

    state, reward, done, _ = envs[i].step(action)
    if done:
        envs[i].random_snakes = random_snakes
        envs[i].random_apples = random_apples

        state = envs[i].reset()
    return i, state, reward

def play():
    for i in range(population_n):
        _, state, reward = step_agent(i)

        states[i] = state
        scores[i] += reward

cdef int batch_size = 100

cdef list size = [6, 24, 3]

cdef int population_n = 500

cdef int width = 20
cdef int height = 20

cdef int window_size = 800

cdef list random_apples = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]
cdef list random_snakes = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]

cdef list population = [NeuralNetwork(size[0], size[1], size[2]) for _ in range(population_n)]
cdef list envs = [Snake_env(width, height, window_size, window_size, 2, random_apples, random_snakes) for _ in range(population_n)]
envs[0] = Snake_env(width, height, window_size, window_size, 2, random_apples, random_snakes, pygame_env=True)

cdef list states = [envs[i].reset() for i in range(population_n)]
cdef list scores = [0 for _ in range(population_n)]

with open("population.pkl", "rb") as file:
    #population = pickle.load(file)

    pass

cdef int gen = -1

cdef list avgScores = []

fig, ax = plt.subplots()

def main():
    global gen
    global batch_size
    global random_apples
    global random_snakes
    global envs
    global population
    global scores
    global avgScores
    global states

    while True:
        random_apples = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]
        random_snakes = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]

        gen += 1
        #batch_size += 1

        ax.clear()
        ax.plot(range(gen), avgScores)  # Example data
        plt.pause(0.0001)

        for _ in range(batch_size):
            play()

            envs[0].render(round(scores[0], 2))

        scores2 = scores

        sorted_indices = np.argsort(scores)[-100:]  # Top 5 indices
        bests = [deepcopy(population[i]) for i in sorted_indices]

        for i in range(len(population)):
            agent = mutatedNeuralNetwork(deepcopy(random.choice(bests)), maxMutation=0.07 / max(math.log(gen + 1, 10), 1))

            population[i] = agent

            states[i] = envs[i].reset()

        bests.reverse()

        for i, agent in enumerate(bests):
            population[i] = agent

        with open("bestAgent.pkl", "wb") as file:
          pickle.dump(population[0], file)

        with open("population.pkl", "wb") as file:
          pickle.dump(population, file)

        print(f"Generarion {gen + 1}")
        print(f"Average score: {np.mean(scores)}")
        print("Best score:", max(scores))

        avgScores.append(np.mean(scores))

        scores = [0 for _ in range(population_n)]