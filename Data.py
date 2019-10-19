class Data:

    def __init__(self, line):
        lineSplitted = line.rstrip().split(',')
        self.x = float(lineSplitted[0])
        self.y = float(lineSplitted[1])
        self.cls = int(lineSplitted[2])

    def inputData(self):
        return [self.x, self.y]

    def correctResult(self):
        if (self.cls == 1):
            return [0, 1]
        elif (self.cls == 2):
            return [1, 0]

    def __repr__(self):
        return '[%s, %s, %s]\n' % (self.x, self.y, self.cls)

    def __str__(self):
        return "[" + self.x + ", " + self.y + ", " + self.cls + "]"
