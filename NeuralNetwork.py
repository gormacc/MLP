import numpy as np

class NeuralNetwork:

    def __init__(self, nodesList, learningRate, useBiases, seed):

        self.nodes = nodesList
        self.numberOfLayers = len(nodesList)
        self.learningRate = learningRate
        self.useBiases = useBiases

        #initialize weights and biases if needed
        np.random.seed(seed) 
        self.weights = []
        self.biases = []
        for i in range(1, self.numberOfLayers):
            self.weights.append(self.initializeWeights(nodesList[i], nodesList[i-1]))
            if useBiases:
                self.biases.append(self.initializeWeights(nodesList[i], 1))

        #set default functions
        self.setMidActivationFun(lambda x: 1 / (1 + np.exp(-x)))
        self.setMidDeactivationFun(lambda x: x * (1 - x))
        self.setEndActivationFun(lambda x: 1 / (1 + np.exp(-x)))
        self.setEndDeactivationFun(lambda x: x * (1 - x))

    def initializeWeights(self, rows, columns):
        return np.matrix(np.random.rand(rows, columns) * 2 - 1)

    def setMidActivationFun(self, activationFunction):
        self.midActivationFun = np.vectorize(activationFunction)

    def setMidDeactivationFun(self, deactivationFunction):
        self.midDeactivationFun = np.vectorize(deactivationFunction)

    def setEndActivationFun(self, activationFunction):
        self.endActivationFun = np.vectorize(activationFunction)

    def setEndDeactivationFun(self, deactivationFunction):
        self.endDeactivationFun = np.vectorize(deactivationFunction)

    def predict(self, inArray):
        #prepare input values
        inputLayer = np.matrix(inArray)
        inputLayer = np.transpose(inputLayer)
        #set layers array
        layers = [inputLayer]
        #calculate middle layers
        for i in range(self.numberOfLayers - 2):
            layers.append(self.calculateLayer(layers[-1], i, self.midActivationFun))
        #calculate last layer
        layers.append(self.calculateLayer(layers[-1], self.numberOfLayers-2, self.endActivationFun))

        return layers

    def train(self, inArray, corrArray):
        #prepare correct values
        correctAnswers = np.matrix(corrArray)
        correctAnswers = np.transpose(correctAnswers)
        #calculate layers values
        layers = self.predict(inArray)
        #calculate error array using error function
        errors = np.subtract(correctAnswers, layers[-1])
        #calculate last weights matrix
        self.recalculateWeights(errors, layers[-1], layers[-2], self.numberOfLayers-2, self.endDeactivationFun)
        #calculate rest weights matrices 
        for i in range(self.numberOfLayers - 2, 0, -1):
            #calculate propagated error
            transposedWeights = np.transpose(self.weights[i])
            errors = np.dot(transposedWeights, errors)
            #calculate weight matrix
            self.recalculateWeights(errors, layers[i], layers[i-1], i-1, self.midDeactivationFun)
        return layers[-1]

    def recalculateWeights(self, errors, currentLayer, prevLayer, index, fun):
        #calculate gradient
        gradient = fun(currentLayer)
        gradient = np.multiply(errors, gradient)
        gradient = np.multiply(gradient, self.learningRate)
        #calculate delta
        transposedPrevLayer = np.transpose(prevLayer)
        deltaWeights = np.dot(gradient, transposedPrevLayer)
        #actualize weights and biases if needed
        self.weights[index] = np.add(self.weights[index], deltaWeights)
        if self.useBiases: 
            self.biases[index] = np.add(self.biases[index], gradient)

    def calculateLayer(self, prevLayer, index, fun):
        #calculate layer value and biases if needed
        layer = np.dot(self.weights[index], prevLayer)
        if self.useBiases:
             layer = np.add(layer, self.biases[index])

        return fun(layer)
