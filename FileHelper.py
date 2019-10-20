import ClassificationData as cdt
import RegressionData as rdt
import tkinter as tk
from tkinter import filedialog

class FileHelper:

    def LoadClassificationData(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        return self.ParseClassificationFile(file_path)

    def ParseClassificationFile(self, filepath):
        data = []
        with open(filepath) as fp:
            line = fp.readline() # skip line x,y,cls
            line = fp.readline()
            while line:
                data.append(cdt.ClassificationData(line))
                line = fp.readline()
        return data

    def LoadRegressionData(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        return self.ParseRegressionFile(file_path)

    def ParseRegressionFile(self, filepath):
        data = []
        with open(filepath) as fp:
            line = fp.readline() # skip line x,y
            line = fp.readline()
            while line:
                data.append(rdt.RegressionData(line))
                line = fp.readline()
        return data
