import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Functions as f
import matplotlib.pyplot as plt

#values to configure

nodes = [2,7,2]
seed = 1
learningRate = 0.1
useBiases = False
trainLoops = 10

midActivationFun = f.sigmoid
endActivationFun = f.sigmoid
midDeactivationFun = f.dsigmoid
endDeactivationFun = f.dsigmoid
lossFun = f.crossEntropy

#data paths

trainDataPath = "C:/Users/macie/Desktop/studia/SieciNeuronowe/MLP/SN_projekt1/classification/data.simple.train.500.csv"
testDataPath = "C:/Users/macie/Desktop/studia/SieciNeuronowe/MLP/SN_projekt1/classification/data.simple.test.500.csv"

#loading data
fileHelper = fh.FileHelper()
data = fileHelper.LoadClassificationData(trainDataPath)
testData = fileHelper.LoadClassificationData(testDataPath)

#initializing neural network
neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
neuralNetwork.setMidActivationFun(midActivationFun)
neuralNetwork.setMidDeactivationFun(midDeactivationFun)
neuralNetwork.setEndActivationFun(endActivationFun)
neuralNetwork.setEndDeactivationFun(endDeactivationFun)

#training
for _ in range(0, trainLoops):
    for d in data:
        neuralNetwork.train(d.inputData(), d.correctResult())

# x = []
# y = []
# c = []

# #visualizaton
# for d in testData:
#     x.append(d.x)
#     y.append(d.y)
#     c.append(d.cls)

# plt.scatter(x,y,c)
# plt.show()