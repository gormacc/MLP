import numpy as np
import NeuralNetwork as nn
import FileHelper as fh

fileHelper = fh.FileHelper()

data = fileHelper.LoadClassificationData()

testData = fileHelper.LoadClassificationData()

neuralNetwork = nn.NeuralNetwork([2, 3, 5, 2])

for _ in range(0, 100):
    for d in data:
        neuralNetwork.train(d.inputData(), d.correctResult())

print('Boom')

for d in testData:
    print(np.transpose(neuralNetwork.predict(d.inputData())), d.correctResult() ,sep=' - ')

# data = fileHelper.LoadRegressionData()

# testData = fileHelper.LoadRegressionData()

# neuralNetwork = nn.NeuralNetwork([1, 3, 5, 1])

# for _ in range(0, 10):
#     for d in data:
#         neuralNetwork.train(d.inputData(), d.correctResult())

# print('Boom')

# for d in testData:
#     print(np.transpose(neuralNetwork.predict(d.inputData())), d.correctResult() ,sep=' - ')

