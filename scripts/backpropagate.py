import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Neural Network Class with Backpropagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for the hidden layer and output layer
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def feedforward(self, X):
        # Input to hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Hidden to output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)

        return self.output

    def backpropagate(self, X, y, learning_rate=0.01):
        # Perform feedforward to get the output
        self.feedforward(X)

        # Compute the error (difference between predicted and actual output)
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Compute the error for the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases for hidden-to-output layer
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # Update weights and biases for input-to-hidden layer
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.backpropagate(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = mean_squared_error(y, self.feedforward(X))
                print(f"Epoch {epoch}, Loss: {loss}")

# Example usage:
if __name__ == "__main__":
    # Example input (X) and output (y) for training the network
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR problem
    y = np.array([[0], [1], [1], [0]])

    # Initialize the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    nn.train(X, y, epochs=1000, learning_rate=0.1)

    # Test the neural network
    print("Predictions after training:")
    print(nn.feedforward(X))
