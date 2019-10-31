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

#regression 

def meanSquareError(c, p):
    0.5 * np.sum(np.square(np.subtract(c,p)))

def meanSquareErrorDerivative(c, p):
    return np.subtract(c, p)

def logCosh(c, p):
    return np.sum(np.log(np.cosh(np.subtract(p, c))))

def logCoshDerivative(c, p):
    return np.tanh(np.subtract(p, c))


#classification

def crossEntropy(c, p):
    return -np.sum( np.multiply( c, np.log(p) ) )

def crossEntropyDerivative(c,p):
    return -np.divide(c,p)

def absoluteError(c, p):
    return np.sum(np.abs(np.subtract(c, p)))

def absoluteErrorDerivative(c, p):
    return (p>=c)*2 - 1