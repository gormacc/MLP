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

errors = []
errSum = 0
#training
for _ in range(0, trainLoops):
    for d in data:
        predicted = neuralNetwork.train(d.inputData(), d.correctResult(nodes[-1]))
        if d.cls != predicted.argmax() + 1:
            errSum = errSum + 1
        errors.append(errSum / (len(errors)+1))

pCls = []

for d in testData:
    p = neuralNetwork.predict(d.inputData())[-1]
    c = p.argmax() + 1
    pCls.append(c)

colors = {1:'r', 2:'g', 3:'b'}
fig, ax = plt.subplots()
for i in range(len(testData)):
    ax.scatter(testData[i].x, testData[i].y,color=colors[pCls[i]])

plt.show()

plt.plot(range(1, len(errors)+1), errors)

plt.show()