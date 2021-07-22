"""
UnitTest  of read_csv module
"""
import os
import unittest
import numpy as np
import pandas as pd
from src.decision_tree import DecisionTree


class ReadCSV(unittest.TestCase):
    """
    UnitTest class of read_csv module
    """

    def self(self):
        """
        something
        """
        self.data = DecisionTree()

    def test_decison_tree(self):
        """
        decision tree function return dataframe,here we check if it is an instance of pandas
        DataFrame
        """
        data = DecisionTree()
        get_data = data.decision_tree()
        self.assertIsInstance(get_data, pd.DataFrame)

    def test_save_results(self):
        """
        save_results function returns the final cv_results
        so we check if those results are close to what we expect
        """

        data = DecisionTree()
        get_results = data.decision_tree()
        expected = 0.7
        np.testing.assert_allclose(get_results.max(0), expected, atol=0.2)

    def test_result_type(self):
        data = DecisionTree()
        get_data = data.save_results()
        self.assertIsInstance(get_data, pd.DataFrame)
