import pygame
import numpy as np
import random
import math

class Snake_env:
  def __init__(self, width, height, snake_length, pygame_env=False):
    if pygame_env:
      pygame.init()

    self.snake_length = snake_length

    self.surface = pygame.display.set_mode((400, 400))

    self.width = width
    self.height = height

    self.snake = None
    self.apple = None

    self.direction = (1, 0)

    self.reset()

  def getInput(self):
    snakePos = self.snake[-1]
    diff = np.radians(snakePos - self.apple)

    x = np.array([math.atan2(diff[0], diff[1]), snakePos[0] / self.width, snakePos[1] / self.height])

    return x

  def getBoard(self):
    try:
      board = np.zeros((self.height, self.width))

      for pos in self.snake:
        if pos[0] > self.height - 1 or pos[0] < 0 or pos[1] > self.width - 1 or pos[1] < 0:

          raise ValueError

        board[pos[0], pos[1]] = 1

      board[self.apple[0], self.apple[1]] = 5

      return board, False

    except Exception as e:
      return np.zeros((self.height, self.width)), True

  def getDistance(self, pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

  def calculateReward(self, oldPos, newPos, apple):
    if tuple(newPos) == apple:
      return 100 * len(self.snake), True  # High reward for eating apple

    if any(np.array_equal(newPos, segment) for segment in self.snake[:-1]) and tuple(self.snake[-1]) != tuple(self.apple):
      return -20, True  # Penalty for self-collision

    old_dist = self.getDistance(oldPos, apple)
    new_dist = self.getDistance(newPos, apple)

    return 0.1 * len(self.snake) if new_dist < old_dist else -0.1, False  # Adjusted rewards

  def step(self, action):
    old_direction = self.direction

    if action == 0:
      self.direction = (0, 1)

    elif action == 1:
      self.direction = (1, 0)

    elif action == 2:
      self.direction = (0, -1)

    elif action == 3:
      self.direction = (-1, 0)

    deltaDirection = np.array(old_direction) - np.array(self.direction)

    done3 = False

    if abs(deltaDirection[0]) == 2 or abs(deltaDirection[1]) == 2:
      done3 = True

    old_pos = self.snake[0]

    pos1 = self.snake[-1]

    self.move()

    pos2 = self.snake[-1]

    if tuple(pos2) == self.apple:
      # Grow the snake by keeping the tail
      new_segment = old_pos

      self.snake = self.snake.tolist()

      self.snake.insert(0, new_segment)

      self.snake = np.array(self.snake)

      self.apple = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))

      while tuple(self.apple) in map(tuple, self.snake):
        self.apple = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))

    pos2 = self.snake[-1]

    board, done = self.getBoard()

    reward, done2 = self.calculateReward(pos1, pos2, self.apple)

    if done2 or done3:
      done = True

    if done:
      reward = -10

    return board, reward, done, []

  def move(self):
    newSnake = []

    for i in range(len(self.snake) - 1):
      nextPos = self.snake[i + 1]

      newSnake.append(nextPos)

    newSnake.append(self.snake[-1] + self.direction)

    self.snake = np.array(newSnake)

  def reset(self):
    self.snake = np.array([[random.randint(self.width // 4, self.width - self.width // 4), random.randint(self.height // 4, self.height - self.height // 4)]])

    self.apple = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))

    while self.apple in self.snake:
      self.apple = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))

    self.direction = (1, 0)

    board, _ = self.getBoard()

    return board

  def render(self):
    self.surface.fill((0, 0, 0))

    pygame.event.get()

    for pos in self.snake:
      pygame.draw.rect(self.surface, (0, 255, 0), pygame.Rect(pos[0] * (400 / self.width), pos[1] * (400 / self.height), 400 / self.width, 400 / self.height))

    pygame.draw.rect(self.surface, (255, 50, 50), pygame.Rect(self.snake[0][0] * (400 / self.width), self.snake[0][1] * (400 / self.height), 400 / self.width, 400 / self.height))

    pygame.draw.rect(self.surface, (255, 0, 0), pygame.Rect(self.apple[0] * (400 / self.width), self.apple[1] * (400 / self.height), 400 / self.width, 400 / self.height))

    pygame.display.flip()