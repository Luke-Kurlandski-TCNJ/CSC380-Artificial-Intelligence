"""
"""

import math

import numpy as np

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

class Neuron:

	def __init__(self, weights=None, output=None):
		self.weights = weights
		self.output = output
	
	def output(self, x):
		return sigmoid(np.dot(self.weights, x))

class NeuralNetwork:

	def __init__(self, n_input, n_hidden, n_output, rnd_min=-.05, rnd_max=.05):
		
		self.learning_rate = .25
		self.epochs = 1000

		self.input_layer = [Neuron() for _ in range(n_input)]
		self.hidden_layer = [Neuron((rnd_max - rnd_min) * np.random.rand(1, n_hidden + 1) + rnd_min) for _ in range(n_hidden)]
		self.output_layer = [Neuron((rnd_max - rnd_min) * np.random.rand(1, n_output + 1) + rnd_min) for _ in range(n_output)]

	def fit(self, X, y):
		
		for x, t in zip(X, y):
			outputs = self.forward(x)
			deltas_output = [o * (1-o) * (t-o) for o in outputs[1]]
			deltas_hidden = []
			for o in outputs[0]:
				s = 0
				
				deltas_hidden.append(o * (1-o) * s)

	def forward(self, x):

		first = [neuron.output(x) for neuron in self.hidden_layer]
		second = [neuron.output(first) for neuron in self.output_layer]
		return np.array([first, second])

	def predict(X):

		pass


def main():
	X = np.array([
		[-1, 0, 0],
		[-1, 0, 1],
		[-1, 1, 0],
		[-1, 1, 1],
	])
	target_and = np.array([0, 0, 0, 1])
	target_or = np.array([0, 1, 1, 1])
	target_xor = np.array([0, 1, 1, 0])
	target_nand = np.array([1, 1, 1, 0])
	target_nor = np.array([1, 0, 0, 0])

	model = NeuralNetwork()
	model.fit(X, target_or)

if __name__ == "__main__":
	main()