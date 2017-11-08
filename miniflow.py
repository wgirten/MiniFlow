class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) to which this Node receives inputs from.
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node will pass a value.
        self.outbound_node = []
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