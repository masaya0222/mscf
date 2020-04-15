import numpy as np
import os


def load_library(libname):
    path = os.path.dirname(__file__)
    return np.ctypeslib.load_library(libname, path)


