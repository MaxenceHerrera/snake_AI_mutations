import numpy as np
import random

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.w1 = np.random.random((hidden_size, input_size)) * 2 - 1

    self.w2 = np.random.random((output_size, hidden_size)) * 2 - 1

    self.x = None

    self.z1 = None
    self.z2 = None

    self.a1 = None
    self.a2 = None

  def softmax(self, x):
    # Subtract the maximum value for numerical stability
    # This prevents potential overflow issues with large exponential values
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

  def relu(self, x):
    return np.where(x > 0, x, 0)

  def relu_derivative(self, x):
    return np.where(x > 0, 1, 0)

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    return x * (1 - x)

  def forward(self, x):
    self.z1 = np.dot(self.w1, x)

    self.a1 = self.relu(self.z1)

    self.z2 = np.dot(self.w2, self.a1)
    self.a2 = self.softmax(self.z2)

    return self.a2

  def chooseAction(self, x):
    return np.argmax(self.forward(x))

class mutatedNeuralNetwork(NeuralNetwork):
  def __init__(self, original, maxMutation=0.25):
    super().__init__(original.input_size, original.hidden_size, original.output_size)

    self.w1 = original.w1
    self.w2 = original.w2

    dw1 = np.random.random((original.hidden_size, original.input_size)) * random.uniform(-maxMutation, maxMutation)
    dw2 = np.random.random((original.output_size, original.hidden_size)) * random.uniform(-maxMutation, maxMutation)

    self.w1 += dw1
    self.w2 += dw2