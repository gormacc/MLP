import numpy as np
import NeuralNetwork as nn
import FileHelper as fh

fileHelper = fh.FileHelper()

# data = fileHelper.LoadClassificationData()
# testData = fileHelper.LoadClassificationData()

# neuralNetwork = nn.NeuralNetwork([2, 3, 5, 2])

# for _ in range(0, 100):
#     for d in data:
#         neuralNetwork.train(d.inputData(), d.correctResult())

# print('Boom')

# for d in testData:
#     print(np.transpose(neuralNetwork.predict(d.inputData())), d.correctResult() ,sep=' - ')

data = fileHelper.LoadRegressionData("C:/Users/macie/Desktop/studia/SieciNeuronowe/MLP/SN_projekt1/regression/data.activation.train.100.csv")
testData = fileHelper.LoadRegressionData("C:/Users/macie/Desktop/studia/SieciNeuronowe/MLP/SN_projekt1/regression/data.activation.train.100.csv")

nodes = [1,3,2,3,1]
isRegression = True
learningRate = 0.1
useBiases = False

neuralNetwork = nn.NeuralNetwork(nodes, isRegression, learningRate, useBiases)

for _ in range(0, 1):
    for d in data:
        neuralNetwork.train(d.inputData(), d.correctResult())

print('Boom')

for d in testData:
    print(np.transpose(neuralNetwork.predict(d.inputData())), d.correctResult() ,sep=' - ')

