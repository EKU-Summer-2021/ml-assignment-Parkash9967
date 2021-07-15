"""
   this file contains function read csv
"""
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split


def fetch_data():
    """
       this function will fetch data
    """
    path = "https://raw.githubusercontent.com/Parkash9967/Test/master/avocado.csv"
    return urllib.request.urlretrieve(path, "avocado.csv")



def load_data():
    """
       this function will laod data
    """
    fetch_data()
    csv_path = "avocado.csv"
    dataset = pd.read_csv(csv_path)
    x_data = dataset[['Small Bags', 'Large Bags', 'Total Volume', '4046', '4225', '4770', 'Large Bags',
                      'Total Bags']]
    target = dataset['AveragePrice']
    input_train, input_test, target_train, target_test = train_test_split \
        (x_data, target, test_size=0.2, random_state=42)
    return input_train, input_test, target_train, target_test
