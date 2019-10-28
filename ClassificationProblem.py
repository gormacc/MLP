import numpy as np
import NeuralNetwork as nn
import FileHelper as fh

#values to configure

nodes = [2,7,2]
seed = 1
learningRate = 0.1
useBiases = False
trainLoops = 100

midActivationFun = lambda x: 1 / (1 + np.exp(-x))
endActivationFun = lambda x: 1 / (1 + np.exp(-x))
midDeactivationFun = lambda x: x * (1 - x)
endDeactivationFun = lambda x: x * (1 - x)
lossFun = lambda c,p : np.subtract(c,p)

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
neuralNetwork.setLossFunction(lossFun)

#training
for _ in range(0, trainLoops):
    for d in data:
        neuralNetwork.train(d.inputData(), d.correctResult())

#visualizaton
for d in testData:
    print(np.transpose(neuralNetwork.predict(d.inputData())[-1]), d.correctResult() ,sep=' - ')