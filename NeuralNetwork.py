import numpy as np

class NeuralNetwork:

    def __init__(self, nodesList, isRegression, learningRate = 0.1, useBiases = True):

        self.nodes = nodesList
        self.numberOfLayers = len(nodesList)
        self.isRegression = isRegression
        self.learningRate = learningRate
        self.useBiases = useBiases

        self.weights = []
        self.biases = []
        for i in range(1, self.numberOfLayers):
            self.weights.append(self.initializeWeights(nodesList[i], nodesList[i-1]))
            if useBiases:
                self.biases.append(self.initializeWeights(nodesList[i], 1))

        #set default functions
        self.setActivationFunction(lambda x: 1 / (1 + np.exp(-x)))
        self.setDeactivationFunction(lambda x: x * (1 - x))

    def initializeWeights(self, rows, columns):
        return np.matrix(np.random.rand(rows, columns) * 2 - 1)

    def setActivationFunction(self, activationFunction):
        self.activationFunction = np.vectorize(activationFunction)

    def setDeactivationFunction(self, deactivationFunction):
        self.deactivationFunction = np.vectorize(deactivationFunction)

    def predict(self, inArray):

        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)

        layers = [inputLayer]

        for i in range(self.numberOfLayers - 2):
            layers.append(self.calculateLayer(layers[-1], i, self.activationFunction))

        layers.append(self.calculateLayer(layers[-1], self.numberOfLayers-2, lambda x : x))

        return layers[-1]

    def train(self, inArray, corrArray):

        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)
        correctAnswers = np.matrix(corrArray)
        correctAnswers = np.transpose(correctAnswers)

        layers = [inputLayer]
        for i in range(self.numberOfLayers - 2):
            layers.append(self.calculateLayer(layers[-1], i, self.activationFunction))

        layers.append(self.calculateLayer(layers[-1], self.numberOfLayers-2, lambda x : x))

        errors = np.subtract(correctAnswers, layers[-1])
        # errors = self.activationFunction(errors)        
        self.recalculateWeights(errors, layers[-1], layers[-2], self.numberOfLayers-2, lambda x : 1)

        for i in range(self.numberOfLayers - 2, 0, -1):
            transposedWeights = np.transpose(self.weights[i])
            errors = np.dot(transposedWeights, errors)

            self.recalculateWeights(errors, layers[i], layers[i-1], i-1, self.deactivationFunction)

    def recalculateWeights(self, errors, currentLayer, prevLayer, index, fun):
        gradient = fun(currentLayer)
        gradient = np.multiply(errors, gradient)
        gradient = np.multiply(gradient, self.learningRate)

        transposedPrevLayer = np.transpose(prevLayer)
        deltaSecondWeights = np.dot(gradient, transposedPrevLayer)

        self.weights[index] = np.add(self.weights[index], deltaSecondWeights)
        if self.useBiases: 
            self.biases[index] = np.add(self.biases[index], gradient)

    def calculateLayer(self, prevLayer, index, fun):
        layer = np.dot(self.weights[index], prevLayer)
        if self.useBiases:
             layer = np.add(layer, self.biases[index])

        return fun(layer)
