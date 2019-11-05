import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Functions as f
import matplotlib.pyplot as plt

# values to configure

# seed = 1
learningRate = 0.001
useBiases = False
trainLoops = 20

seeds = [253, 124, 951, 536, 938]

# loading data
fileHelper = fh.FileHelper()
trainData = fileHelper.LoadClassificationData()
testData = fileHelper.LoadClassificationData()

midActivationFun = f.sigmoid
endActivationFun = f.linear
midDeactivationFun = f.dsigmoid
endDeactivationFun = f.dlinear
lossFun = f.meanSquareErrorDerivative

errors = [0,0,0,0,0,0,0,0,0,0,0]

nodesList = [
    [2, 1],
    [2, 5, 1],
    [2, 10, 1],
    [2, 5, 10, 1],
    [2, 10, 5, 1],
    [2, 5, 10, 5, 1],
    [2, 10, 5, 10, 1],
    [2, 10, 5, 10, 5, 1],
    [2, 5, 10, 5, 10, 1],
    [2, 5, 10, 10, 5, 1],
    [2, 10, 5, 5, 10, 1]
]

for j in range (0, len(seeds)):
    seed = seeds[j]
    for i in range (0, len(nodesList)):
    # initializing neural network
        nodes = nodesList[i]

        neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
        neuralNetwork.setMidActivationFun(midActivationFun)
        neuralNetwork.setMidDeactivationFun(midDeactivationFun)
        neuralNetwork.setEndActivationFun(endActivationFun)
        neuralNetwork.setEndDeactivationFun(endDeactivationFun)
        neuralNetwork.setLossFunction(lossFun)

        # training  
        for _ in range(0, trainLoops):
            for d in trainData:
                predicted = neuralNetwork.train(
                    d.inputData(), d.correctResult())

        sumOfErrors = 0
        
        for d in testData:
            p = neuralNetwork.predict(d.inputData())[-1]
            diffSquare = (p.item(0) - d.y) ** 2
            sumOfErrors = sumOfErrors + diffSquare
        
        errors[i] = errors[i] + sumOfErrors/len(testData)

for i in range(0, len(errors)):
    errors[i] = errors[i]/len(seeds)

print(np.round(errors, 2))