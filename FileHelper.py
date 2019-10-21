import ClassificationData as cdt
import RegressionData as rdt
import tkinter as tk
from tkinter import filedialog

class FileHelper:

    def LoadClassificationData(self, filePath = None):
        root = tk.Tk()
        root.withdraw()

        if filePath == None:
            filePath = filedialog.askopenfilename()
        return self.ParseClassificationFile(filePath)

    def ParseClassificationFile(self, filepath):
        data = []
        with open(filepath) as fp:
            line = fp.readline() # skip line x,y,cls
            line = fp.readline()
            while line:
                data.append(cdt.ClassificationData(line))
                line = fp.readline()
        return data

    def LoadRegressionData(self, filePath = None):
        root = tk.Tk()
        root.withdraw()

        if filePath == None:
            filePath = filedialog.askopenfilename()
        return self.ParseRegressionFile(filePath)

    def ParseRegressionFile(self, filepath):
        data = []
        with open(filepath) as fp:
            line = fp.readline() # skip line x,y
            line = fp.readline()
            while line:
                data.append(rdt.RegressionData(line))
                line = fp.readline()
        return data
