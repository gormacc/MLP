import numpy as np

class Data:

    def __init__(self, line):
        lineSplitted = line.rstrip().split(',')
        self.x = float(lineSplitted[0])
        self.y = float(lineSplitted[1])
        self.cls = int(lineSplitted[2])

    def inputData(self):
        return [self.x, self.y]

    def correctResult(self):
        result = np.zeros(2).astype(int)
        result[self.cls - 1] = 1
        return result

    def __repr__(self):
        return '[%s, %s, %s]\n' % (self.x, self.y, self.cls)

    def __str__(self):
        return "[" + self.x + ", " + self.y + ", " + self.cls + "]"
