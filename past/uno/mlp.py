import numpy as np

# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Define the derivative of the ReLU activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Define the inputs and outputs for the neural network
inputs = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

outputs = np.array([[0],
                    [1],
                    [1],
                    [0]])

# Set the seed for reproducibility
np.random.seed(1)

# Initialize the weights randomly
weights_0 = 2 * np.random.random((3, 4)) - 1
weights_1 = 2 * np.random.random((4, 1)) - 1

# Define the number of iterations
num_iterations = 60000

# Set the learning rate
learning_rate = 0.1

# Train the neural network
for i in range(num_iterations):

    # Forward propagation
    layer_0 = inputs
    layer_1 = relu(np.dot(layer_0, weights_0))
    layer_2 = relu(np.dot(layer_1, weights_1))

    # Calculate the error
    layer_2_error = outputs - layer_2

    # Print the mean absolute error every 10000 iterations
    if i % 10000 == 0:
        print("Error: " + str(np.mean(np.abs(layer_2_error))))

    # Backpropagation
    layer_2_delta = layer_2_error * relu_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weights_1.T)
    layer_1_delta = layer_1_error *  _derivative(layer_1)

    # Update the weights
    weights_1 += learning_rate * layer_1.T.dot(layer_2_delta)
    weights_0 += learning_rate * layer_0.T.dot(layer_1_delta)

# Print the final predictions
print("Predictions:")
print(layer_2)
