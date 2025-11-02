from snake_env import Snake_env
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

batch_size = 100

size = [6, 24, 3]

population_n = 500

width = 20
height = 20

window_size = [600, 600]


random_apples = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]
random_snakes = [(random.randint(1, width - 2), random.randint(1, height - 2)) for _ in range(10000)]

population = [NeuralNetwork(size[0], size[1], size[2]) for _ in range(population_n)]
envs = [Snake_env(width, height, window_size[0], window_size[1], 2, random_apples, random_snakes) for _ in range(population_n)]
envs[0] = Snake_env(width, height, window_size[0], window_size[1], 2, random_apples, random_snakes, pygame_env=True)

states = [envs[i].reset() for i in range(population_n)]
scores = [0 for _ in range(population_n)]

with open("population.pkl", "rb") as file:
    population = pickle.load(file)

    pass

gen = -1

avgScores = []

fig, ax = plt.subplots()

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

        envs[0].render(population[0], population[0].forward2(envs[0].getInput()), round(scores[0], 2))

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
