"""
Main class
"""

import numpy as np
import pandas as pd

from src import Polynomial
from src.decision_tree import DecisionTree
from src.svm import SupportVectorMachines

if __name__ == '__main__':
    coeffs = np.array([1, 0, 0])
    polynom = Polynomial(coeffs)
    print(polynom.evaluate(3))
    print(polynom.roots())

# d = DecisionTree()
# d.decision_tree()
# d.save_results()
dataset = pd.read_csv('https://raw.githubusercontent.com/Parkash9967/Test/master/heart.csv')

s = SupportVectorMachines(dataset[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                          'oldpeak', 'slope', 'ca', 'thal']],
                          dataset['target'])
print(s.s_v_m())
print(s.save_result_csv())
s.plotting_svm()
