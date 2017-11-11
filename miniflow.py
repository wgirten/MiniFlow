import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) to which this Node receives inputs from.
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node will pass a value.
        self.outbound_nodes = []
        # For each inbound node, add this node as an outbound Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # Each Node will calculate a value that represents its output.
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Calculates the output value, given the values from the inbound Nodes.
        """
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # An Input Node will have no inbound Nodes.
        Node.__init__(self)

    def forward(self, value=None):
        # Since an Input Node has no inbound Nodes,
        # it will not have to calculate an output value.
        if value is not None:
            self.value = value


class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 0  # initialize output value
        for n in self.inbound_nodes:
            self.value += n.value


class Multiply(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = 0  # initialize output value
        for n in self.inbound_nodes:
            self.value *= n.value


class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b


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
        m = y.shape[0]
        self.value = np.sum((1. / m) * np.square((y - a)))


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


def forward_pass(graph):
    """
    Performs a forward pass through a list of sorted nodes.

    :param graph: A topologically sorted list of nodes.
    :return: None
    """
    for n in graph:
        n.forward()
