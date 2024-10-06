
import numpy as np
## Start with input
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        #initialise weights and bias
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
    # Sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Sig deriv
    def deriv_sigmoid(self, x):
        return x * (1 - x)




## Need code for a node

## Weights

## Am I Biased?

## Output

## GET BACK

