from copy import deepcopy
import pygame
import numpy as np
import random
import math

class Snake_env:
  def __init__(self, width, height, renderWidth, renderHeight, snake_length, random_apples, random_snakes, pygame_env=False):
    self.rendering = False

    if pygame_env:
      pygame.init()
      pygame.font.init()

      self.font = pygame.font.SysFont('Arial', 30)

      self.rendering = True

    self.snake_length = snake_length

    self.surface = pygame.display.set_mode((renderWidth, renderHeight))

    self.width = width
    self.height = height
    self.renderWidth = renderWidth
    self.renderHeight = renderHeight

    self.random_apples = random_apples
    self.random_snakes = random_snakes

    self.deaths = 0

    self.snake = None
    self.apple = None

    self.direction = (1, 0)

    self.reset()

    self.action_number = 0

    self.last4Positions = []
    self.last4Actions = np.zeros(20).tolist()

  def get_relative_tiles(self, board):
      x, y = self.snake[-1]
      dx, dy = self.direction  # e.g. (1,0), (0,1), (-1,0), (0,-1)

      # relative offsets: forward, left, right, backward
      if (dx, dy) == (1, 0):  # facing right
          offsets = [(1, 0), (0, -1), (0, 1), (-1, 0)]
      elif (dx, dy) == (0, 1):  # facing down
          offsets = [(0, 1), (1, 0), (-1, 0), (0, -1)]
      elif (dx, dy) == (-1, 0):  # facing left
          offsets = [(-1, 0), (0, 1), (0, -1), (1, 0)]
      elif (dx, dy) == (0, -1):  # facing up
          offsets = [(0, -1), (-1, 0), (1, 0), (0, 1)]

      tiles = []
      for ox, oy in offsets:
          tx, ty = x + ox, y + oy
          if 0 <= tx < self.width and 0 <= ty < self.height:
              tiles.append(board[tx, ty])
          else:
              tiles.append(-1)  # treat out-of-bounds as wall
      return tiles

  def getInput(self):
    snakePos = self.snake[-1]
    diff = snakePos - self.apple

    diff = np.degrees(diff)

    distance = np.linalg.norm(diff) / self.width

    xPos = snakePos[0] / self.width
    yPos = snakePos[1] / self.height

    angle_snake = math.atan2(self.direction[1], self.direction[0])
    angle_to_apple = math.atan2(diff[1], diff[0])
    angle_diff = angle_to_apple - angle_snake
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]

    board, _ = self.getBoard()

    snakePos = deepcopy(snakePos)

    if (snakePos[0] >= board.shape[0] - 1):
        snakePos[0] -= 1

    if (snakePos[1] >= board.shape[1] - 1):
        snakePos[1] -= 1

    tile1, tile2, tile3, tile4 = self.get_relative_tiles(board)

    x = [
            math.cos(angle_diff),
            math.sin(angle_diff),
            distance,
            tile1,
            tile2,
            tile3,
        ]

    x = np.array(
        x, # + self.last4Actions,
        dtype=np.float32)

    return x

  def getBoard(self):
    try:
      board = np.zeros((self.height, self.width))

      for pos in self.snake:
        if pos[0] > self.height - 1 or pos[0] < 0 or pos[1] > self.width - 1 or pos[1] < 0:

          raise ValueError

        board[pos[0], pos[1]] = 1

      board[0][:] = -1
      board[-1][:] = -1
      board[:, 0] = -1
      board[:, -1] = -1

      board[self.apple[0], self.apple[1]] = 0

      return board, False

    except Exception as e:
      return np.zeros((self.height, self.width)), True

  def getDistance(self, pos1, pos2):
    diff = pos1 - pos2

    return math.sqrt((diff[0]*diff[0] + diff[1]*diff[1]))

  def calculateReward(self, oldPos, newPos, apple, dead):
    if (dead):
        return -5, True

    if tuple(newPos) == apple:
      return 5 / math.log(self.action_number + 50, 50), False  # High reward for eating apple

    if any(np.array_equal(newPos, segment) for segment in self.snake[:-1]) and tuple(self.snake[-1]) != tuple(self.apple):
      return -7, True  # Penalty for self-collision

    return 0.01, False  # Adjusted rewards

  def step(self, action):
    old_direction = self.direction

    self.action_number += 1

    if action == 0:  # turn left
        if self.direction == (1, 0):  # right → up
            self.direction = (0, -1)
        elif self.direction == (0, -1):  # up → left
            self.direction = (-1, 0)
        elif self.direction == (-1, 0):  # left → down
            self.direction = (0, 1)
        elif self.direction == (0, 1):  # down → right
            self.direction = (1, 0)

    elif action == 2:  # turn right
        if self.direction == (1, 0):  # right → down
            self.direction = (0, 1)
        elif self.direction == (0, 1):  # down → left
            self.direction = (-1, 0)
        elif self.direction == (-1, 0):  # left → up
            self.direction = (0, -1)
        elif self.direction == (0, -1):  # up → right
            self.direction = (1, 0)

    old_pos = self.snake[0]

    pos1 = self.snake[-1]

    self.move()

    pos2 = self.snake[-1]

    board, doneBoard = self.getBoard()

    reward, done = self.calculateReward(pos1, pos2, self.apple, doneBoard)

    if tuple(pos2) == self.apple:
      self.last4Positions = []

      # Grow the snake by keeping the tail
      new_segment = old_pos

      self.snake = self.snake.tolist()

      self.snake.insert(0, new_segment)

      self.snake = np.array(self.snake)

      i = len(self.snake)

      self.apple = self.random_apples[i]

      testing = True

      while testing:
          i += 1

          if (i >= len(self.random_apples)):
              self.apple = [random.randint(1, self.width - 2), random.randint(1, self.height - 2)]

          else:
              self.apple = self.random_apples[i]

          for pos in self.snake:
              if pos.tolist() != self.apple:
                  testing = False

                  break

    if (pos2.tolist() in self.last4Positions):
        reward -= 1

    if (len(self.last4Positions) >= 20):
        self.last4Positions.pop(0)

    self.last4Actions.pop(0)
    self.last4Actions.append(action - 1)

    self.last4Positions.append(pos2.tolist())

    return board, reward, done, []

  def move(self):
    newSnake = self.snake[1:].tolist()

    newSnake.append(self.snake[-1] + self.direction)

    self.snake = np.array(newSnake)

  def reset(self):
    self.last4Actions = np.zeros(20).tolist()
    self.deaths += 1

    i = self.deaths

    self.action_number = 0
    self.last4Positions = []

    self.snake = np.array([self.random_snakes[i]])

    self.apple = self.random_apples[i]

    while self.apple in self.snake:
        i += 1

        if (i >= len(self.random_apples)):
            self.apple = (random.randint(1, self.width - 2), random.randint(1, self.height - 2))

        else:
            self.apple = self.random_apples[i]

    self.direction = (1, 0)

    board, _ = self.getBoard()

    return board

  def render(self, agent="", probs=[[0, 0, 0]], score=0):
    self.surface.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    for pos in self.snake:
      pygame.draw.rect(self.surface, (0, 255, 0), pygame.Rect(pos[0] * (self.renderWidth / self.width), pos[1] * (self.renderHeight / self.height), self.renderWidth / self.width, self.renderHeight / self.height))

    if (agent != ""):
        size = self.renderHeight / 2.5
        start = self.renderWidth * 0.1
        span = self.renderWidth * 0.5

        gap1 = size / agent.w1.shape[1]
        gap2 = size / agent.w2.shape[1]
        gap3 = size / agent.w2.shape[0]

        neuronSize = min(gap1, gap2, gap3) / 3
        gap = min(gap1, gap2, gap3)
        maxSize = max(agent.w1.shape[1], agent.w2.shape[1], agent.w2.shape[0]) * gap + self.renderHeight * 0.1


        half1 = agent.w1.shape[1] / 2 * gap
        half2 = agent.w2.shape[1] / 2 * gap
        half3 = agent.w2.shape[0] / 2 * gap

        for i, w1 in enumerate(agent.w1):
            pos = (start + span / 2, maxSize / 2 - half2 + (i * gap))

            pygame.draw.circle(self.surface, np.array([255, 255, 255]) * agent.sigmoid(probs[1][i]), pos, neuronSize)

            for j, w2 in enumerate(w1):
                pos1 = (start, maxSize / 2 - half1 + (j * gap))

                color = (255, 0, 0) if w2 < 0 else (0, 0, 255)

                pygame.draw.line(self.surface, color, pos, pos1, int(math.log(abs(w2 * 5) + 2, 2)))

        for i, w1 in enumerate(agent.w2):
            pos = (start + span, maxSize / 2 - half3 + (i * gap))

            pygame.draw.circle(self.surface, np.array([255, 255, 255]) * agent.sigmoid(probs[2][i]), pos, neuronSize)

            for j, w2 in enumerate(w1):
                pos1 = (start + span / 2, maxSize / 2 - half2 + (j * gap))

                color = (255, 0, 0) if w2 < 0 else (0, 0, 255)

                pygame.draw.line(self.surface, color, pos, pos1, int(math.log(abs(w2 * 5) + 2, 2)))

        for i in range(agent.w1.shape[1]):
            pos = (start, maxSize / 2 - half1 + (i * gap))

            pygame.draw.circle(self.surface, np.array([255, 255, 255]) * agent.sigmoid(probs[0][i]), pos, neuronSize)


    pygame.draw.rect(self.surface, (255, 50, 50), pygame.Rect(self.snake[-1][0] * (self.renderWidth / self.width), self.snake[-1][1] * (self.renderHeight / self.height), self.renderWidth / self.width, self.renderHeight / self.height))

    pygame.draw.rect(self.surface, (255, 0, 0), pygame.Rect(self.apple[0] * (self.renderWidth / self.width), self.apple[1] * (self.renderHeight / self.height), self.renderWidth / self.width, self.renderHeight / self.height))

    text_surface = self.font.render(f"Current score: {score}", True, (255, 255, 255))  # White text
    text_surface2 = self.font.render(f"Current direction: {self.direction}", True, (255, 255, 255))  # White text
    self.surface.blit(text_surface, (30, 30))

    pygame.display.flip()
