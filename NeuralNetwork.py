import numpy as np

class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        self.firstWeights = self.initializeWeights(self.hiddenNodes, self.inputNodes)
        self.secondWeights = self.initializeWeights(self.outputNodes, self.hiddenNodes)

        self.firstBias = self.initializeWeights(self.hiddenNodes, 1)
        self.secondBias = self.initializeWeights(self.outputNodes, 1)

        self.setLearningRate()

    def initializeWeights(self, rows, columns):
        return np.matrix(np.random.rand(rows, columns) * 2 - 1)

    def setLearningRate(self, learningRate = 0.2):
        self.learningRate = learningRate

    def predict(self, inArray):

        sigmoid = np.vectorize(lambda x : 1 / (1 + np.exp(-x)))

        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)

        hiddenLayer = np.dot(self.firstWeights, inputLayer)
        hiddenLayer = np.add(hiddenLayer, self.firstBias)
        hiddenLayer = sigmoid(hiddenLayer)

        outputLayer = np.dot(self.secondWeights, hiddenLayer)
        outputLayer = np.add(outputLayer, self.secondBias)
        outputLayer = sigmoid(outputLayer)

        return outputLayer


    def train(self, inArray, corrArray):

        sigmoid = np.vectorize(lambda x : 1 / (1 + np.exp(-x)))
        dsigmoid = np.vectorize(lambda x : x * (1 - x))

        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)
        correctAnswers = np.matrix(corrArray)
        correctAnswers = np.transpose(correctAnswers)

        hiddenLayer = np.dot(self.firstWeights, inputLayer)
        hiddenLayer = np.add(hiddenLayer, self.firstBias)
        hiddenLayer = sigmoid(hiddenLayer)

        outputLayer = np.dot(self.secondWeights, hiddenLayer)
        outputLayer = np.add(outputLayer, self.secondBias)
        outputLayer = sigmoid(outputLayer)

        secondErrors = np.subtract(correctAnswers, outputLayer)
        #secondErrors = np.transpose(secondErrors)  #??

        secondGradient = dsigmoid(outputLayer)
        secondGradient = np.dot(secondErrors, secondGradient)
        secondGradient = np.multiply(secondGradient, self.learningRate)

        transposedHiddenLayer = np.transpose(hiddenLayer)
        deltaSecondWeights = np.dot(secondGradient, transposedHiddenLayer)

        self.secondWeights = np.add(self.secondWeights, deltaSecondWeights)
        self.secondBias = np.add(self.secondBias, secondGradient)

        transposedSecondWeights = np.transpose(self.secondWeights)
        firstErrors = np.dot(transposedSecondWeights, secondErrors)

        firstGradient = dsigmoid(hiddenLayer)
        firstGradient = np.multiply(firstErrors, firstGradient)
        firstGradient = np.multiply(firstGradient, self.learningRate)

        transposedInputLayer = np.transpose(inputLayer)
        deltaFirstWeights = np.dot(firstGradient, transposedInputLayer)

        self.firstWeights = np.add(self.firstWeights, deltaFirstWeights)
        self.firstBias = np.add(self.firstBias, firstGradient)




