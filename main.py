import numpy as np
import NeuralNetwork as nn


valDictionary = { (0,0) : [0], (0,1) : [1] , (1,0) : [1], (1,1) : [0] }
trueDictionary = {True : 1, False : 0}

neuralNetwork = nn.NeuralNetwork([2, 3, 5, 1])

for _ in range(0, 10000):
    firstBit = 0 + np.random.rand() < 0.5
    secondBit = 0 + np.random.rand() < 0.5
    neuralNetwork.train([trueDictionary[firstBit], trueDictionary[secondBit]], valDictionary[(firstBit, secondBit)] )

print(neuralNetwork.predict([0,0]))
print(neuralNetwork.predict([1,0]))
print(neuralNetwork.predict([0,1]))
print(neuralNetwork.predict([1,1]))
