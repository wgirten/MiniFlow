import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) to which this Node receives inputs from.
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node will pass a value.
        self.outbound_nodes = []
        # Each Node will calculate a value that represents its output.
        self.value = None
        # Keys are Inputs to this Node. Values are the partial derivatives
        # of this Node with respect to that Input.
        self.gradients = {}
        # For each inbound node, add this node as an outbound Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        """
        Forward propagation.

        Calculates the output value, given the values from the inbound Nodes.
        """
        raise NotImplementedError

    def backward(self):
        """
        Backward propagation.
        """
        raise NotImplementedError


class Input(Node):
    """
    A generic input Node to a neural network.
    """
    def __init__(self):
        # An Input Node will have no inbound Nodes.
        Node.__init__(self)

    def forward(self, value=None):
        # Since an Input Node has no inbound Nodes,
        # it will not have to calculate an output value.
        if value is not None:
            self.value = value

    def backward(self):
        # An Input has no Inputs, so the derivative is 0.
        self.gradients = {self: 0}
        # Weights and biases may be inputs, so sum gradient
        # from output gradients
        for n in self.outbound_nodes:
            gradient_cost = n.gradients[self]
            self.gradients[self] += gradient_cost * 1


class Linear(Node):
    """
    A Node that performs a Linear Transform.
    """
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Perform a linear transform.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values
        """
        # Initialize a partial derivative for each of the inbound nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # For each of the output nodes, the gradient will change depending on each
        # output. Therefore, gradients are summed over each output.
        for n in self.outbound_nodes:
            # Get the partial derivative of the cost with respect to this Node.
            gradient_cost = n.gradients[self]
            # Set the partial derivative of the cost with respect to this Node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(gradient_cost, self.inbound_nodes[1].value.T)
            # Set the partial derivative of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, gradient_cost)
            # Set the partial derivative of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(gradient_cost, axis=0, keepdims=False)



class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        Calculates the Sigmoid function.

        :param x: A numpy array object.
        :return: The result of the sigmoid function.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize a partial derivative for each of the inbound nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        # For each of the output nodes, the gradient will change depending on each
        # output. Therefore, gradients are summed over each output.
        for n in self.outbound_nodes:
            # Get the partial derivative of the cost with respect to this Node.
            gradient_cost = n.gradients[self]
            sigmoid = self.value
            x = self.inbound_nodes[0]
            self.gradients[x] += sigmoid * (1 - sigmoid) * gradient_cost


class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # Re-shape both arrays to be (3,1) and ensures element-wise subtraction as expected.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.value = np.mean(np.square(y - a))

    def backward(self):
        """
        Calculates the gradient of the cost of the network.
        """
        # This is the final Node of the network, so outbound nodes are not a concern.
        y = self.inbound_nodes[0]
        a = self.inbound_nodes[1]
        y_value = self.inbound_nodes[0].value.reshape(-1, 1)
        a_value = self.inbound_nodes[1].value.reshape(-1, 1)
        error = y_value - a_value
        m = self.inbound_nodes[0].value.shape[0]
        self.gradients[y] = (2 / m) * error
        self.gradients[a] = (-2 / m) * error


def topological_sort(feed_dict):
    """
    Sorts generic Nodes in topological order using Kahn's Algorithm.

    :param feed_dict: A dictionary where the key is a `Input` node and the
                      value is the respective value feed to that node.
    :return: A list of topologically sorted Nodes.
    """
    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward and backward pass through a list of sorted nodes.

    :param graph: A topologically sorted list of nodes.
    :return: None
    """
    # Perform a forward pass
    for n in graph:
        n.forward()

    # Perform a backward pass
    for n in graph[::-1]:
        n.backward()
