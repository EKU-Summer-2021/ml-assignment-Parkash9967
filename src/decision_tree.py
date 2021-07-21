"""
file of Decision Tree module
"""
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


class DecisionTree:
    """
    class of Decision Tree module
    """

    def __str__(self):
        return self.__class__.__name__

    def __init__(self):
        self.model = DecisionTreeRegressor()

    def decision_tree(self):
        """
        function of Decision Tree module to predict score
        """
        data = pd.read_csv('https://raw.githubusercontent.com/Parkash9967/Test/master/avocado.csv')
        data.drop('Unnamed: 0', inplace=True, axis=1)  # pylint: disable=E1101
        data.drop('year', inplace=True, axis=1)  # pylint: disable=E1101
        data_x = data[[  # pylint: disable= E1136
            'Small Bags', 'Large Bags', 'Total Volume', 'Total Bags', '4046', '4225', '4770', 'Large Bags',  # pylint: disable= E1136
            'XLarge Bags']]  # pylint: disable= E1136
        data_y = data['AveragePrice']  # pylint: disable= E1136
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
        dt_model = DecisionTreeRegressor(random_state=10)
        dt_model.fit(x_train, y_train)
        y_pred = dt_model.predict(x_test)
        pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        dt_model.score(x_train, y_train)

        # model = DecisionTreeRegressor()
        gird_search = GridSearchCV(self.model,
                                   param_grid={'min_samples_split': [7, 15, 30], 'min_samples_leaf': [2, 4, 6, 8]},
                                   cv=5,
                                   n_jobs=1,
                                   return_train_score=True)
        gird_search.fit(x_train, y_train)
        cv_rests = gird_search.cv_results_
        self.model.fit(x_train, y_train)
        self.model.predict(x_test)
        save = pd.DataFrame({'score': cv_rests["mean_test_score"],
                             'parameters': cv_rests["params"]})
        plt.scatter(y_test, y_pred)
        plt.savefig("plot.png")
        plt.show()
        return save

    def save_results(self):
        """
                function of Decision Tree module to save results into saprate directory
        """
        final_result_csv = self.decision_tree()
        path_dir = os.getcwd()
        directory = 'Results'
        directory_dt = 'DR'
        # path_dir = r'C:\Users\Parkash Ladhani\PycharmProjects\ml-assignment-Parkash9967\Results'
        path_results = path_dir + r'\Results'
        path_dr = path_dir + r'\Results\DR'
        if not os.path.exists(path_results):
            os.mkdir(os.path.join(path_dir, directory))
        if not os.path.exists(path_dr):
            os.mkdir(os.path.join(path_results, directory_dt))
        file_name = 'csv_resutls' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        file_location = os.path.join(path_dr, file_name)
        final_result_csv.to_csv(file_location, index=False)
        return final_result_csv
