import numpy as np
import random
from snake_env import Snake_env
from neuralNetwork import NeuralNetwork, mutatedNeuralNetwork

def play(population):
  for i, agent in enumerate(population):
    action = np.argmax(agent.forward(envs[i].getInput()))

    state, reward, done, _ = envs[i].step(action)

    if done:
      state = envs[i].reset()
      scores[i] -= 10

    states[i] = state
    scores[i] += reward

batch_size = 100

size = [3, 140, 4]

population_n = 1000

population = [NeuralNetwork(size[0], size[1], size[2]) for _ in range(population_n)]
envs = [Snake_env(10, 10, 2) for _ in range(population_n)]
envs[0] = Snake_env(10, 10, 2, pygame_env=True)

states = [envs[i].reset() for i in range(population_n)]
scores = [0 for _ in range(population_n)]

while True:
  for _ in range(batch_size):
    play(population)

    envs[0].render()

  scores2 = scores

  bests = []

  sorted_indices = np.argsort(scores)[-250:][::-1]  # Top 5 indices
  bests = [population[i] for i in sorted_indices]

  for i, agent in enumerate(population):
    agent = mutatedNeuralNetwork(random.choice(bests), maxMutation=0.01)

    population[i] = agent

    states[i] = envs[i].reset()

  for i, agent in enumerate(bests):
    population[i] = agent

  print("New Gen")
  print(f"Average score: {np.mean(scores)}")
  print("Best score:", max(scores))

  scores = [0 for _ in range(population_n)]