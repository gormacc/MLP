import numpy as np


class NeuralNetwork:

    def __init__(self, nodesList):

        self.nodes = nodesList
        self.numberOfLayers = len(nodesList)

        self.weights = []
        self.biases = []
        for i in range(1, self.numberOfLayers):
            self.weights.append(self.initializeWeights(nodesList[i], nodesList[i-1]))
            self.biases.append(self.initializeWeights(nodesList[i], 1))

        self.setLearningRate()

    def initializeWeights(self, rows, columns):
        return np.matrix(np.random.rand(rows, columns) * 4 - 2)

    def setLearningRate(self, learningRate=0.2):
        self.learningRate = learningRate

    def predict(self, inArray):

        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)

        layers = [inputLayer]

        for i in range(self.numberOfLayers - 1):
            layers.append(self.calculateLayer(layers[-1], i, sigmoid))

        return layers[-1]

    def train(self, inArray, corrArray):

        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        dsigmoid = np.vectorize(lambda x: x * (1 - x))

        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)
        correctAnswers = np.matrix(corrArray)
        correctAnswers = np.transpose(correctAnswers)

        layers = [inputLayer]
        for i in range(self.numberOfLayers - 1):
            layers.append(self.calculateLayer(layers[-1], i, sigmoid))

        errors = np.subtract(correctAnswers, layers[-1])
        # secondErrors = np.transpose(lastErrors)  #??

        self.recalculateWeights(errors, layers[-1], layers[-2], self.numberOfLayers-2, dsigmoid)

        for i in range(self.numberOfLayers - 2, 0, -1):
            transposedWeights = np.transpose(self.weights[i])
            errors = np.dot(transposedWeights, errors)

            self.recalculateWeights(errors, layers[i], layers[i-1], i-1, dsigmoid)

    def recalculateWeights(self, errors, currentLayer, prevLayer, index, fun):
        gradient = fun(currentLayer)
        gradient = np.multiply(errors, gradient)
        gradient = np.multiply(gradient, self.learningRate)

        transposedPrevLayer = np.transpose(prevLayer)
        deltaSecondWeights = np.dot(gradient, transposedPrevLayer)

        self.weights[index] = np.add(self.weights[index], deltaSecondWeights)
        self.biases[index] = np.add(self.biases[index], gradient)

    def calculateLayer(self, prevLayer, index, fun):
        layer = np.dot(self.weights[index], prevLayer)
        layer = np.add(layer, self.biases[index])
        return fun(layer)
