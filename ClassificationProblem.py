import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Functions as f
import matplotlib.pyplot as plt

#values to configure

seed = 1
learningRate = 0.1
useBiases = True
trainLoops = 30

midActivationFun = f.sigmoid
endActivationFun = f.sigmoid
midDeactivationFun = f.dsigmoid
endDeactivationFun = f.dsigmoid
lossFun = f.meanSquareErrorDerivative

#loading data
fileHelper = fh.FileHelper()
trainData = fileHelper.LoadClassificationData()
testData = fileHelper.LoadClassificationData()

maxCls = max(trainData, key= lambda x: x.cls).cls
nodes = [2,1,maxCls]

#initializing neural network
neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
neuralNetwork.setMidActivationFun(midActivationFun)
neuralNetwork.setMidDeactivationFun(midDeactivationFun)
neuralNetwork.setEndActivationFun(endActivationFun)
neuralNetwork.setEndDeactivationFun(endDeactivationFun)
neuralNetwork.setLossFunction(lossFun)

errors = []
errSum = 0
#training
for _ in range(0, trainLoops):
    for d in trainData:
        predicted = neuralNetwork.train(d.inputData(), d.correctResult(nodes[-1]))
        if d.cls != predicted.argmax() + 1:
            errSum = errSum + 1
        errors.append(errSum / (len(errors)+1))

pCls = []
trainCls = []

for d in testData:
    p = neuralNetwork.predict(d.inputData())[-1]
    c = p.argmax() + 1
    pCls.append(c)

for d in trainData:
    p = neuralNetwork.predict(d.inputData())[-1]
    c = p.argmax() + 1
    trainCls.append(c)

colors = {1:'red', 2:'green', 3:'blue', 4:'yellow'}
fig, ax = plt.subplots()
for i in range(len(testData)):
    ax.scatter(testData[i].x, testData[i].y,color=colors[pCls[i]])
plt.title("Klasyfikacja na zbiorze testowym")

fig2, ax2 = plt.subplots()
for i in range(len(trainData)):
    ax2.scatter(trainData[i].x, trainData[i].y,color=colors[trainCls[i]])
plt.title("Klasyfikacja na zbiorze treningowym")

fig2, ax2 = plt.subplots()
for i in range(len(testData)):
    if (pCls[i] != testData[i].cls):
        ax2.scatter(testData[i].x, testData[i].y,color='red')
    else: 
        ax2.scatter(testData[i].x, testData[i].y,color='green')
plt.title("Błędy na zbiorze testowym")

fig2, ax2 = plt.subplots()
for i in range(len(trainData)):
    if (trainCls[i] != trainData[i].cls):
        ax2.scatter(trainData[i].x, trainData[i].y,color='red')
    else: 
        ax2.scatter(trainData[i].x, trainData[i].y,color='green')
plt.title("Błędy na zbiorze treningowym")

plt.figure()
plt.plot(range(1, len(errors)+1), errors)
plt.title("Errors over train times")

plt.show()