'''Write a python program to show Back Propagation Network for XOR function with Binary Input
and Output'''



import numpy as np
_________________________________________________________________________________________________________________________________________________________________

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
_________________________________________________________________________________________________________________________________________________________________
# Define the neural network class
class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.rand(2, 2)
        self.bias_input_hidden = np.random.rand(1, 2)
        self.weights_hidden_output = np.random.rand(2, 1)
        self.bias_hidden_output = np.random.rand(1, 1)

    def feedforward(self, X):
        # Forward propagation
        self.hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output

    def backpropagation(self, X, y, learning_rate):
        # Backpropagation
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        # Training loop
        for epoch in range(epochs):
            # Forward propagation
            output = self.feedforward(X)
            # Backpropagation
            self.backpropagation(X, y, learning_rate)
            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}: Loss = {loss}")

_________________________________________________________________________________________________________________________________________________________________

# Define the input data and labels for XOR function
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network instance
nn = NeuralNetwork()

# Train the neural network
epochs = 10000
learning_rate = 0.1
nn.train(X, y, epochs, learning_rate)

# Test the trained model
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.feedforward(test_data)
print("Predictions:")
print(predictions)

_________________________________________________________________________________________________________________________________________________________________

