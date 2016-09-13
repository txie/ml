"""
 source: http://iamtrask.github.io/2015/07/12/basic-python-network/
"""

import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if deriv is True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[0, 0, 1, 1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1

for iter in xrange(10):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    print 'l1_error'
    print l1_error, '\n'
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1

    print 'nolin(l1, True)'
    print nonlin(l1, True), '\n'
    l1_delta = l1_error * nonlin(l1, True)

    print 'l1_delta'
    print l1_delta, '\n'

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

    print 'syn0'
    print syn0, '\n'

print "Output After Training:"
print l1
