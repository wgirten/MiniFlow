"""
Builds and runs a graph with MiniFlow.
"""
import numpy as np
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

# Define inputs, weights, and biases matrices
X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict2 = {X: X_, W: W_, b: b_}
graph2 = topological_sort(feed_dict2)
output2 = forward_pass(f, graph2)

"""
Output should be:
[[-9., 4.],
[-9., 4.]]
"""
print('Linear Node output:')
print(output2)