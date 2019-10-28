class RegressionData:

    def __init__(self, line):
        lineSplitted = line.rstrip().split(',')
        self.x = float(lineSplitted[0])
        self.y = float(lineSplitted[1])

    def inputData(self):
        return [self.x]

    def correctResult(self):
        return [self.y]

    def __repr__(self):
        return '[%s, %s]\n' % (self.x, self.y)

    def __str__(self):
        return "[" + self.x + ", " + self.y + "]"