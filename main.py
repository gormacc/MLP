import numpy as np
import NeuralNetwork as nn
import FileHelper as fh
import Data as dt

fileHelper = fh.FileHelper()

data = fileHelper.LoadData()

testData = fileHelper.LoadData()

neuralNetwork = nn.NeuralNetwork([2, 3, 5, 2])

for _ in range(0, 100):
    for d in data:
        neuralNetwork.train(d.inputData(), d.correctResult())

print('Boom')

for d in testData:
    print(np.transpose(neuralNetwork.predict(d.inputData())), d.correctResult() ,sep=' - ')

# print(neuralNetwork.predict([0,0]))
# print(neuralNetwork.predict([1,0]))
# print(neuralNetwork.predict([0,1]))
# print(neuralNetwork.predict([1,1]))
