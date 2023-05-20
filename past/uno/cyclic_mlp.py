import numpy as np
import random as rng
from copy import deepcopy


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def init_network():
    adjacency_fwd = [[[None], [None], [None], [None], [None]],
                    [[None], [None], [None], [None], [None]],
                    [[None], [None], [None], [None], [None]],
                    [[None], [None], [None], [None], [None]]]
    adjacency_back =    [[[None], [None], [None], [None], [None]],
                        [[None], [None], [None], [None], [None]],
                        [[None], [None], [None], [None], [None]],
                        [[None], [None], [None], [None], [None]]]
    active_nodes =  [[1, None, None, None, None],
                    [1, None, None, None, None],
                    [1, None, None, None, None],
                    [1, None, None, None, None]]
    outgoing_weights =   [[[1], [None], [None], [None], [None]],
                         [[1], [None], [None], [None], [None]],
                         [[1], [None], [None], [None], [None]],
                         [[1], [None], [None], [None], [None]]]
    classifier_weights =    [[None, None, None, None, None],
                            [None, None, None, None, None],
                            [None, None, None, None, None],
                            [None, None, None, None, None]]
    return adjacency_fwd, adjacency_back, outgoing_weights, classifier_weights, active_nodes

def connect_node(start_node, end_node, adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights):
    x1, y1 = start_node
    x2, y2 = end_node

    start_is_input, first_connection = False, False
    if y2 == 0:
        print("Connecting to input node not allowed.")
        return
    if y1 == 0:
        start_is_input = True

    # Update adjacency matrices
    if adjacency_fwd[x1][y1] == [None]:
        first_connection = True
        adjacency_fwd[x1][y1] = [[x2, y2]]
    else:
        adjacency_fwd[x1][y1].append([x2, y2])
    if adjacency_back[x2][y2] == [None]:
        adjacency_back[x2][y2] = [[x1, y1]]
    else:
        adjacency_back[x2][y2].append([x1, y1])

    # end_node's first connection
    if active_nodes[x2][y2] == None:
        active_nodes[x2][y2] = 1
        classifier_weights[x2][y2] = rng.uniform(0, 1)

    if start_is_input and first_connection:
        outgoing_weights[x1][y1][0] = 1 # Only weights of 1 are allowed out of input layer
    elif start_is_input and not first_connection:
        outgoing_weights[x1][y1].append([1])
    elif not start_is_input and first_connection:
        outgoing_weights[x1][y1][0] = rng.uniform(0, 1)
    else:
        outgoing_weights[x1][y1].append(rng.uniform(0, 1))

def forward(input, adjacency_fwd, outgoing_weights, classifier_weights):
    forward_values = [  [input[0].item(), 0, 0, 0, 0],
                        [input[1].item(), 0, 0, 0, 0],
                        [input[2].item(), 0, 0, 0, 0],
                        [input[3].item(), 0, 0, 0, 0]]

    outgoing_shape = [len(outgoing_weights), len(outgoing_weights[0])]
    x1, y1 = outgoing_shape[0], outgoing_shape[1]
    for x2 in range(x1):
        for y2 in range(y1):
            neighbors = adjacency_fwd[x2][y2]
            if neighbors != [None]:
                input_node = False
                if y2 == 0:
                    input_node = True
                for i, [x3, y3] in enumerate(neighbors):
                    if input_node:
                        forward_values[x3][y3] += outgoing_weights[x2][y2][i] * forward_values[x2][y2]
                    else:
                        forward_values[x3][y3] += relu(outgoing_weights[x2][y2][i] * forward_values[x2][y2])

    # Classification Node
    F = np.array(forward_values)
    CW = np.array(classifier_weights)
    CW[CW == None] = 0
    output_signal = (F*CW).sum()
    return output_signal

def back_prop(expected_output, output, active_nodes, outgoing_weights, classifier_weights, adjacency_fwd, adjacency_back, learning_rate=1):
    total_error = expected_output - output
    A = np.array(active_nodes)
    A[A == None] = 0
    OW = np.array(classifier_weights)
    OW[OW == None] = 0
    weighted_error = (A*OW)/((A*OW).sum()) * total_error

    reversed_back = deepcopy(adjacency_back)
    reversed_back.reverse()
    for row in reversed_back:
        row.reverse()

    for x1, nodes in enumerate(reversed_back):
        for y1, neighbors in enumerate(nodes):
            if neighbors != [None]:
                x_original, y_original = (OW.shape[0]-1)-x1, (OW.shape[1]-1)-y1  # Indexing of shape is one less than shape value
                node_weights = []
                for x2, y2 in neighbors:
                    i = adjacency_fwd[x2][y2].index([x_original, y_original])
                    node_weights.append(outgoing_weights[x2][y2][i])
                classifier_weight = classifier_weights[x_original][y_original]  # -1 due to input layer weights
                node_weights.append(classifier_weight)
                node_weights_numpy = np.array(node_weights)
                total_loss = weighted_error[x_original][y_original]
                node_losses = node_weights_numpy / node_weights_numpy.sum() * total_loss
                for i2, [x2, y2] in enumerate(neighbors):
                    i = adjacency_fwd[x2][y2].index([x_original, y_original])
                    if y_original != 0:
                        outgoing_weights[x2][y2][i] += learning_rate * relu_derivative(node_losses[i2])
                    else:
                        outgoing_weights[x2][y2][i] += learning_rate * node_losses[i2]
                classifier_weights[x_original][y_original] += learning_rate * relu_derivative(node_losses[-1])

def generate_valid_neighbors(node_idx, outgoing_shape):
    x1, y1 = outgoing_shape[0], outgoing_shape[1]
    x2, y2 = node_idx
    if y2 == 0:
        return [[x2, y2+1]]
    else:
        valid_neighbors = []
        potential_neighbors = [[x2+1, y2], [x2-1, y2], [x2, y2+1], [x2, y2-1]]
        for x3, y3 in potential_neighbors:
            if x1 > x3 >= 0 and y1 > y3 > 0:
                valid_neighbors.append([x3, y3])
        return valid_neighbors

def fully_connect(adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights):
    outgoing_shape = np.array(outgoing_weights).shape
    x1, y1 = outgoing_shape[0], outgoing_shape[1]
    for x2 in range(x1):
        for y2 in range(y1):
                    neighbors = generate_valid_neighbors([x2, y2], [outgoing_shape[0], outgoing_shape[1]])
                    for neighbor in neighbors:
                        connect_node([x2, y2], neighbor, adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)

def main():
    adjacency_fwd, adjacency_back, outgoing_weights, classifier_weights, active_nodes = init_network()

    # TESTING PHASE 1
    # DATA 1:
    # inputs = np.array([[[1], [1], [0], [0]]])
    # expected_outputs = [5]
    # DATA 2:
    inputs = np.array([[[1], [1], [0], [0]],
                       [[1], [0], [1], [0]],
                       [[0], [0], [0], [1]],
                       [[1], [0], [0], [1]]])
    expected_outputs = [3, 5, 8, 9]
    # CONNECTION 1: Two connected nodes.
    # connect_node([0, 0], [0, 1], adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)
    # connect_node([0, 1], [0, 2], adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)
    # CONNECTION 2: Fully connected matrix.
    fully_connect(adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)
    # CONNECTION 3:
    # connect_node([0, 0], [0, 1], adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)
    # connect_node([1, 0], [1, 1], adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)
    # connect_node([2, 0], [2, 1], adjacency_fwd, adjacency_back, active_nodes, outgoing_weights, classifier_weights)

    # Training loop
    epochs = 200
    learning_rate = .1
    avg_errors = []
    for i in range(epochs):
        error = 0
        for i2, input in enumerate(inputs):
            expected_output = expected_outputs[i2]
            output = forward(input, adjacency_fwd, outgoing_weights, classifier_weights)
            back_prop(expected_output, output, active_nodes, outgoing_weights, classifier_weights, adjacency_fwd, adjacency_back, learning_rate=learning_rate)
            error += abs(output - expected_output)
            if i == epochs-1:
                print()
                print('Yhat' + str(i2) + ': ' + str(output))
                print('Y' + str(i2) + ': ' + str(expected_output))
        avg_errors.append(abs(error/len(inputs)))
        print(abs(error/len(inputs)))

if __name__ == "__main__":
    main()