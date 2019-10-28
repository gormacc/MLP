import numpy as np


#activation and derivatives

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1-np.tanh(x)**2

def relu(x):
    if x > 0:
        return x
    else:
        return 0

def drelu(x):
    if x > 0:
        return 1
    else:
        return 0

def linear(x):
    return x

def dlinear(x):
    return 1


#loss functions

def differenceError(c, p):
    return np.subtract(c, p)

#regression 

def meanSquareError(c, p):
    diff = differenceError(c,p)
    power = np.power(diff, 2)
    sumV = np.sum(power)
    return sumV / len(c)

def absoluteError(c, p):
    diff = differenceError(c,p)
    absolute = np.abs(diff)
    sumV = np.sum(absolute)
    return sumV / len(c)

def smoothAbsoluteError(c, p):
    diff = differenceError(c,p)
    absolute = np.abs(diff)
    sum = 0
    for elem in absolute:
        if elem < 1:
            sum += 0.5 * np.power(elem, 2)
        else:
            sum += (elem - 0.5)
    return sum

#classification

def crossEntropy(c, p):
    epsilon = 1e-12
    p = np.clip(p, epsilon, 1. - epsilon)
    N = p.shape[0]
    ce = -np.sum( np.multiply( c, np.log(p) ) ) / N
    return ce