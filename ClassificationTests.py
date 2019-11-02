import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Functions as f
import matplotlib.pyplot as plt

# values to configure

forLoops = 10

# seed = 1
learningRate = 0.1
useBiases = False
trainLoops = 100

seeds = [253, 124, 951, 536, 938, 2, 908, 254, 400, 481]
midActivationFunList = [f.sigmoid, f.tanh, f.relu, f.linear]
endActivationFunList = [f.sigmoid, f.tanh, f.sigmoid, f.sigmoid]
midDeactivationFunList = [f.dsigmoid, f.dtanh, f.drelu, f.dlinear]
endDeactivationFunList = [f.dsigmoid, f.dtanh, f.dsigmoid, f.dsigmoid]
lossFun = f.meanSquareErrorDerivative

# data paths

trainDataPath = "C:/Users/Karol/Desktop/ProjOB/SN/SN_projekt1/classification/data.simple.train.500.csv"
testDataPath = "C:/Users/Karol/Desktop/ProjOB/SN/SN_projekt1/classification/data.simple.test.500.csv"

# loading data
fileHelper = fh.FileHelper()
# data = fileHelper.LoadClassificationData(trainDataPath)
# testData = fileHelper.LoadClassificationData(testDataPath)
trainData = fileHelper.LoadClassificationData()
testData = fileHelper.LoadClassificationData()

maxCls = max(trainData, key=lambda x: x.cls).cls
nodes = [2, 6, maxCls]
errors = [0,0,0,0]
for j in range (0, forLoops):
    seed = seeds[j]
    for i in range (0, len(midActivationFunList)):
    # initializing neural network
        midActivationFun = midActivationFunList[i]
        endActivationFun = endActivationFunList[i]
        midDeactivationFun = midDeactivationFunList[i]
        endDeactivationFun = endActivationFunList[i]

        neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
        neuralNetwork.setMidActivationFun(midActivationFun)
        neuralNetwork.setMidDeactivationFun(midDeactivationFun)
        neuralNetwork.setEndActivationFun(endActivationFun)
        neuralNetwork.setEndDeactivationFun(endDeactivationFun)
        neuralNetwork.setLossFunction(lossFun)

        errSum = 0
        # training  
        for _ in range(0, trainLoops):
            for d in trainData:
                predicted = neuralNetwork.train(
                    d.inputData(), d.correctResult(nodes[-1]))

        sumOfErrors = 0
        
        for d in testData:
            p = neuralNetwork.predict(d.inputData())[-1]
            c = p.argmax() + 1
            if (c != d.cls):
                sumOfErrors = sumOfErrors + 1
        
        errors[i] = errors[i] + sumOfErrors/len(testData)

for i in range(0, len(errors)):
    errors[i] = errors[i]/forLoops

print(seeds)
print(np.round(errors, 2))