import numpy as np

class ClassificationData:

    def __init__(self, line):
        lineSplitted = line.rstrip().split(',')
        self.x = float(lineSplitted[0])
        self.y = float(lineSplitted[1])
        self.cls = int(lineSplitted[2])

    def inputData(self):
        return [self.x, self.y]

    def correctResult(self, vectorLength):
        result = np.zeros(vectorLength).astype(int)
        result[self.cls - 1] = 1
        return result

    def __repr__(self):
        return '[%s, %s, %s]\n' % (self.x, self.y, self.cls)

    def __str__(self):
        return '[%s, %s, %s]\n' % (self.x, self.y, self.cls)
