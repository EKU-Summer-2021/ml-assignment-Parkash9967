"""
UnitTest  of read_csv module
"""

import unittest
import pandas as pd
from src import read_csv


class ReadCSV(unittest.TestCase):
    """
    UnitTest class of read_csv module
    """

    def test_fetching_data(self):
        """
        UnitTest function of decision tree module
        """
        # input_train, input_test, target_train, target_test = read_csv.load_data()
        dataset = read_csv.fetch_data()
        data = pd.DataFrame(dataset)
        self.assertIsInstance(data, pd.DataFrame)

    def test_loading_data(self):
        """
        UnitTest function of decision tree module
        """
        # input_train, input_test, target_train, target_test = read_csv.load_data()
        dataset = read_csv.load_data()
        data = pd.DataFrame(dataset)
        self.assertIsInstance(data, pd.DataFrame)
