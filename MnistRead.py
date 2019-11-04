from mnist import MNIST
import random
import os
import NeuralNetwork as nn
import Functions as f
import pickleSerializer as ps
from tkinter import filedialog

mnData = MNIST("data")
testImages, testLabels = mnData.load_testing()

#normalize
for i in range(0, len(testImages)):
    for j in range(0, len(testImages[i])):
        testImages[i][j] = testImages[i][j]/255.0

#read file
filePath = filedialog.askopenfilename()
neuralNetwork = ps.deserialize(filePath)
print("deserialized : " + filePath)
#checkCorrectPredictions
correct = 0
for i in range(0, len(testImages)):
    p = neuralNetwork.predict(testImages[i])[-1]
    if p.argmax() == testLabels[i]:
        correct += 1
print("result : " + str(correct) + "/" + str(i+1))

