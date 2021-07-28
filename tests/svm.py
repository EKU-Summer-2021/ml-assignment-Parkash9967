"""
Unit test module for SVM
"""
import unittest
import pandas as pd
import numpy as np
from src.svm import SupportVectorMachines


class MyTestCase(unittest.TestCase):
    """
    Unit test class for SVM
    """

    def setUp(self):
        dataset = pd.read_csv('https://raw.githubusercontent.com/Parkash9967/Test/master/heart.csv')
        param_grid = {'C': [5, 10], 'kernel': ('linear', 'rbf')}
        self.data = SupportVectorMachines(
            dataset[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                     'oldpeak', 'slope', 'ca', 'thal']],
            dataset['target'], param_grid)

    def test_s_v_m(self):
        """
        SupportVectorMachine function return dataframe,here we check if it is an instance of pandas
        DataFrame
        """
        get_data = self.data.s_v_m()
        self.assertIsInstance(get_data, pd.DataFrame)

    def test_save_result_csv(self):
        """
        here we check if the result is close to our expected value
        """
        get_results = self.data.save_result_csv()
        expected = 0.7
        np.testing.assert_allclose(get_results.max(0), expected, atol=0.2)
