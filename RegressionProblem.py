import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Functions as f
import matplotlib.pyplot as plt

#values to configure

nodes = [1,13,1]
seed = 1
learningRate = 0.001
useBiases = False
trainLoops = 100

midActivationFun = f.sigmoid
endActivationFun = f.linear
midDeactivationFun = f.dsigmoid
endDeactivationFun = f.dlinear
lossFun = f.smoothAbsoluteError

#data paths

trainDataPath = "C:/Users/macie/Desktop/studia/SieciNeuronowe/MLP/SN_projekt1/regression/data.activation.train.100.csv"
testDataPath = "C:/Users/macie/Desktop/studia/SieciNeuronowe/MLP/SN_projekt1/regression/data.activation.test.100.csv"

#loading data
fileHelper = fh.FileHelper()
# data = fileHelper.LoadRegressionData(trainDataPath)
# testData = fileHelper.LoadRegressionData(testDataPath)
data = fileHelper.LoadRegressionData()
testData = fileHelper.LoadRegressionData()

#initializing neural network
neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
neuralNetwork.setMidActivationFun(midActivationFun)
neuralNetwork.setMidDeactivationFun(midDeactivationFun)
neuralNetwork.setEndActivationFun(endActivationFun)
neuralNetwork.setEndDeactivationFun(endDeactivationFun)

errors = []

#training
for _ in range(0, trainLoops):
    for d in data:
        predictedValue = neuralNetwork.train(d.inputData(), d.correctResult())
        errors.append(np.abs( d.y - predictedValue.item(0) ) )

#visualizaton

x = []
y = []
yp = []

for d in sorted(testData, key = lambda td: td.x):
    x.append(d.x)
    y.append(d.y)
    yp.append(neuralNetwork.predict(d.inputData())[-1].item(0))

plt.figure(1)
plt.plot(x, yp, 'ob', label = 'Predicted data')
plt.plot(x, y, 'r', label = 'Correct data')
plt.title('Prediction')
plt.legend()

plt.figure(2)
plt.plot(range(1, len(errors)+1), errors)
plt.title('Errors over train times')
plt.show()
