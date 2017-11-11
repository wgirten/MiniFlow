"""
Builds and runs a graph with MiniFlow.
"""
import numpy as np
from miniflow import *

# Calculate Mean Squared Error (MSE)
y, a = Input(), Input()
cost = MSE(y, a)

y_ = np.array([1, 2, 3])
a_ = np.array([4.5, 5, 10])

feed_dict = {y: y_, a: a_}
graph = topological_sort(feed_dict)
forward_pass(graph)

"""
Expected cost: 23.4166666667
"""
print("Network Cost: {}".format(cost.value))
