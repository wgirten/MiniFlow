"""
Builds and runs a graph with MiniFlow.
"""
from miniflow import *

# Define 2 Input Nodes
x, y, z = Input(), Input(), Input()

# Define an Add Node with the 2 above Input Nodes being the input
add = Add(x, y, z)

# The value of x and y Input Nodes
feed_dict = {x: 4, y: 5, z: 10}

# Sort the nodes using topological sort
graph = topological_sort(feed_dict=feed_dict)
output = forward_pass(add, graph)

print("{} + {} + {} = {} (according to MiniFlow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))

# Define 3 Input Nodes which will create a Linear Node
inputs, weights, bias = Input(), Input(), Input()

linear = Linear(inputs, weights, bias)

feed_dict2 = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph2 = topological_sort(feed_dict2)
output2 = forward_pass(linear, graph2)

print("Linear Node output: {}".format(output2))