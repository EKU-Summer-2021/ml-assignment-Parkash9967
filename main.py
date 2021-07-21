"""
Main class
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from src import Polynomial
from src import decision_tree
from src.decision_tree import DecisionTree

if __name__ == '__main__':
    coeffs = np.array([1, 0, 0])
    polynom = Polynomial(coeffs)
    print(polynom.evaluate(3))
    print(polynom.roots())

d = DecisionTree()
d.decision_tree()
d.save_results()

