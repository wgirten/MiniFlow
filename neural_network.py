"""
Builds and runs a graph with MiniFlow.
"""
from miniflow import *

# Define 2 Input Nodes
x, y = Input(), Input()

# Define an Add Node with the 2 above Input Nodes being the input
add = Add(x, y)

# The value of x and y Input Nodes
feed_dict = {x: 10, y: 20}

# Sort the nodes using topological sort
sorted_nodes = topological_sort(feed_dict=feed_dict)
output = forward_pass(add, sorted_nodes)

print("{} + {} = {} (according to MiniFlow)".format(feed_dict[x], feed_dict[y], output))