import numpy as np
import random as rng


def ReLU(input):
    return max(0, input)

def init_network():
    adjacency_fwd = [[[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]],
                    [[0], [0], [0], [0], [0]]]
    adjacency_back =    [[[0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0]],
                        [[0], [0], [0], [0], [0]]]
    active_nodes =  [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]
    outgoing_weights =   [[[1], [0], [0], [0], [0]],
                         [[1], [0], [0], [0], [0]],
                         [[1], [0], [0], [0], [0]],
                         [[1], [0], [0], [0], [0]]]
    classifier_weights =    [[rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)],
                            [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)],
                            [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)],
                            [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)]]
    return adjacency_fwd, adjacency_back, outgoing_weights, classifier_weights, active_nodes

def forward(input, adjacency, weights, output_weights):
    forward_values = [  [input[0].item(), 0, 0, 0, 0],
                        [input[1].item(), 0, 0, 0, 0],
                        [input[2].item(), 0, 0, 0, 0],
                        [input[3].item(), 0, 0, 0, 0]]

    # Iterate over input layer
    input_height = 4
    for x1 in range(input_height):
        y1 = 0
        neighbors = adjacency[x1][y1]
        if neighbors[0] != 0:
            for y3, [x2, y2] in enumerate(neighbors):
                forward_values[x2][y2] += weights[x1][0][y3] * forward_values[x1][0]

    # Iterate over remaining nodes
    for x1, nodes in enumerate(adjacency):
        for y1, neighbors in enumerate(nodes):
            if neighbors[0] != 0 and y1 != 0:
                for y3, [x2, y2] in enumerate(neighbors):
                    forward_values[x2][y2] += ReLU(weights[x1][y1][y3] * forward_values[x1][y1])

    # Classification Node
    F = np.array(forward_values)[:,1:]  # Get rid of input layer
    out_W = np.array(output_weights)
    output_signal = (F*out_W).sum()

    return output_signal

def back(expected_output, output, active_nodes, outgoing_weights, classifier_weights, adjacency_back, learning_rate=1):
    total_error = output - expected_output
    # TESTING
    total_error = 0.5
    outgoing_weights[0][0] = [1]
    outgoing_weights[0][1] = [0.6]
    classifier_weights[0][0] = 0.75
    classifier_weights[0][1] = 0.25
    # TESTING
    A = np.array(active_nodes)
    OW = np.array(classifier_weights)
    weighted_error = (A*OW)/((A*OW).sum()) * total_error
    zeros_column = np.zeros((weighted_error.shape[0], 1))
    weighted_error = np.concatenate((zeros_column, weighted_error), axis=1)  # We include input layer weights

    adjacency_backprop = adjacency_back.copy()
    adjacency_backprop.reverse()
    for row in adjacency_backprop:
        row.reverse()

    max_x, max_y = 4, 4
    for x1, nodes in enumerate(adjacency_backprop):
        for y1, neighbors in enumerate(nodes):
            if neighbors[0] != 0:
                x_original, y_original = max_x-x1-1, max_y-y1+1  # -1 due to indexing, +1 due to weights including input layer,
                node_weights = []
                for y3, [x2, y2] in enumerate(neighbors):  # CONTINUE HERE: Need to ensure neighbor weights are pointing towards current node being updated
                    node_weights.append(outgoing_weights[x2][y2][y3])
                outgoing_weight = classifier_weights[x_original][y_original - 1]  # -1 due to input layer weights
                node_weights.append(outgoing_weight)
                weighted_node_weights = np.array(node_weights)
                total_loss = learning_rate * classifier_weights[x_original][y_original]
                node_losses = weighted_node_weights / weighted_node_weights.sum() * total_loss
                for y3, [x2, y2] in enumerate(neighbors):
                    outgoing_weights[x2][y2][y3] -= node_losses[y3]
                classifier_weights[x_original][y_original] -= node_losses[-1]

    print('test')


def main():
    input = np.array([[1], [1], [0], [0]])
    expected_output = 5
    adjacency_fwd, adjacency_back, weights, output_weights, active_nodes = init_network()

    # Custom testing
    adjacency_fwd[0][0] = [[0, 1]]
    adjacency_back[0][0] = [[0, 0]]
    active_nodes[0][1-1] = 1
    adjacency_fwd[0][1] = [[0, 2]]
    if weights[0][1] == [0]:
        weights[0][1][0] = rng.uniform(0, 1)
    else:
        weights[0][1].append(rng.uniform(0, 1))
    adjacency_back[0][1] = [[0, 1]]
    active_nodes[0][2-1] = 1
    output = forward(input, adjacency_fwd, weights, output_weights)
    back(expected_output, output, active_nodes, weights, output_weights, adjacency_back)


if __name__ == "__main__":
    main()