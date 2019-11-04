from mnist import MNIST
import random
import os
import NeuralNetwork as nn
import Functions as f
import pickleSerializer as ps

#load mnist data
mnData = MNIST("data")
trainImages, trainLabels = mnData.load_training()
testImages, testLabels = mnData.load_testing()

#normalize
for i in range(0, len(trainImages)):
    for j in range(0, len(trainImages[i])):
        trainImages[i][j] = trainImages[i][j]/255.0

for i in range(0, len(testImages)):
    for j in range(0, len(testImages[i])):
        testImages[i][j] = testImages[i][j]/255.0

#serialization path
serializePath = "neuralNetworksPickles/500_300HU_1S_B_001LR_MSE_SIG_NORM_"
#parameters for neural network
nodes = [784, 500, 300, 10]
seed = 1
learningRate = 0.01
useBiases = True
trainLoops = 100

midActivationFun = f.sigmoid
endActivationFun = f.sigmoid
midDeactivationFun = f.dsigmoid
endDeactivationFun = f.dsigmoid
lossFun = f.meanSquareErrorDerivative
#initialization of neural network
neuralNetwork = nn.NeuralNetwork(nodes, learningRate, useBiases, seed)
neuralNetwork.setMidActivationFun(midActivationFun)
neuralNetwork.setMidDeactivationFun(midDeactivationFun)
neuralNetwork.setEndActivationFun(endActivationFun)
neuralNetwork.setEndDeactivationFun(endDeactivationFun)
neuralNetwork.setLossFunction(lossFun)

for tl in range(trainLoops):
    #train
    for i in range(0, len(trainImages)):
        iCls = [0] * 10
        iCls[trainLabels[i]] = 1
        neuralNetwork.train(trainImages[i], iCls)
    #serialize
    filePath = serializePath + str(tl) + "TL.pickle"
    ps.serialize(filePath, neuralNetwork)
    print("serialized : " + filePath)
    #check correct predictions
    correct = 0
    for i in range(0, len(testImages)):
        p = neuralNetwork.predict(testImages[i])[-1]
        if p.argmax() == testLabels[i]:
            correct += 1
    print("tested TL" + str(tl) + " : " + str(correct) + "/" + str(i+1))

