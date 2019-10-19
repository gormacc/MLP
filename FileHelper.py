import Data as dt
import tkinter as tk
from tkinter import filedialog

class FileHelper:

    def LoadData(self):
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename()
        return self.ParseFile(file_path)

    def ParseFile(self, filepath):
        data = []
        with open(filepath) as fp:
            line = fp.readline() # skip line x,y,cls
            line = fp.readline()
            while line:
                data.append(dt.Data(line))
                line = fp.readline()
        return data
