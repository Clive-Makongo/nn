
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

    
    # Activation
    def forward(self, X):
        self.activations = [X]

        print(f"Input: {X}")

        for i in range(len(self.weights)):
            #net = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            prev_activation = self.activations[-1]
            net = np.dot(prev_activation, self.weights[i]) + self.biases[i]
            print(f"Layer {i+1} pre-activation: {net}")
            
            #self.activations.append(self.sigmoid(net))
            activation = self.sigmoid(net)
            print(f"Layer {i+1} activation: {activation}")
            
            self.activations.append(activation)

        
        return self.activations[-1]

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        delta = self.activations[-1] - y          # Calc Error

        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
            
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db


## Weights X

## Am I Biased? X

## Output

## GET BACK

