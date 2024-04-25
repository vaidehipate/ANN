'''Implement Artificial Neural Network training process in Python by using Forward Propagation,
Back Propagation'''

______________________________________________________________________________________________________________________________________________________________


import numpy as np
______________________________________________________________________________________________________________________________________________________________

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
______________________________________________________________________________________________________________________________________________________________

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize weights and biases randomly
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_input_hidden = np.random.rand(1, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden_output = np.random.rand(1, output_size)
        self.learning_rate = learning_rate

    def feedforward(self, X):
        # Forward propagation
        self.hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output

    def backpropagation(self, X, y):
        # Backpropagation
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        # Training loop
        for epoch in range(epochs):
            # Forward propagation
            output = self.feedforward(X)
            # Backpropagation
            self.backpropagation(X, y)
            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}: Loss = {loss}")
______________________________________________________________________________________________________________________________________________________________
# Define the input data and labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network instance
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# Train the neural network
epochs = 1000
nn.train(X, y, epochs)

# Test the trained model
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.feedforward(test_data)
print("Predictions:")
print(predictions)

______________________________________________________________________________________________________________________________________________________________
