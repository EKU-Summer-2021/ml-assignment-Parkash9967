"""
Main class
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from src import Polynomial
from src import read_csv


from src.dt_module import DecisionTree

print(read_csv.load_data())