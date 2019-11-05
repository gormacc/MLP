import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Functions as f
import matplotlib.pyplot as plt

#values to configure

nodes = [1,10,10,1]
seed = 1
learningRate = 0.001
useBiases = True
trainLoops = 30

midActivationFun = f.sigmoid
endActivationFun = f.linear
midDeactivationFun = f.dsigmoid
endDeactivationFun = f.dlinear
lossFun = f.meanSquareErrorDerivative

#loading data
fileHelper = fh.FileHelper()
trainData = fileHelper.LoadRegressionData()
testData = fileHelper.LoadRegressionData()

#initializing neural network
neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
neuralNetwork.setMidActivationFun(midActivationFun)
neuralNetwork.setMidDeactivationFun(midDeactivationFun)
neuralNetwork.setEndActivationFun(endActivationFun)
neuralNetwork.setEndDeactivationFun(endDeactivationFun)
neuralNetwork.setLossFunction(lossFun)

errors = []

#training
for _ in range(0, trainLoops):
    for d in trainData:
        predictedValue = neuralNetwork.train(d.inputData(), d.correctResult())
        errors.append(np.abs( d.y - predictedValue.item(0) ) )

#visualizaton

x = []
y = []
xtrain = []
ytrain = []
yp = []
yptrain = []

for d in sorted(testData, key = lambda td: td.x):
    x.append(d.x)
    y.append(d.y)
    yp.append(neuralNetwork.predict(d.inputData())[-1].item(0))

for d in sorted(trainData, key = lambda td: td.x):
    xtrain.append(d.x)
    ytrain.append(d.y)
    yptrain.append(neuralNetwork.predict(d.inputData())[-1].item(0))

plt.figure(1)
plt.plot(x, yp, 'ob', label = 'Predicted data')
plt.plot(x, y, 'r', label = 'Correct data')
plt.title('Prediction on test set')
plt.legend()

plt.figure(2)
plt.plot(xtrain, yptrain, 'ob', label = 'Predicted data')
plt.plot(xtrain, ytrain, 'r', label = 'Correct data')
plt.title('Prediction on train set')
plt.legend()

plt.figure(3)
plt.plot(range(1, len(errors)+1), errors)
plt.title('Errors over train times')
plt.show()
