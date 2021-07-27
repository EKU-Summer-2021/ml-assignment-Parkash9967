"""
file of Support Vector Machines
"""
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


class SupportVectorMachines:  # pylint: disable= R0902
    """
    class  Support Vector Machines  classification
    """

    def __str__(self):
        return self.__class__.__name__

    def __init__(self, x_data, y_data):
        """
            constructor with parameters   of  Support Vector Machines class
        """
        self.x_data = x_data
        self.y_data = y_data
        self.clf = None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=0.2)
        self.svc_model = svm.SVC(kernel='linear')

    def s_v_m(self):
        """
             function svm for  predicting score and gird search
        """

        self.svc_model.fit(self.x_train, self.y_train)
        # data = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
        score = self.svc_model.score(self.x_train, self.y_train)
        print(score)
        parameters = {'kernel': ('linear', 'rbf'), 'C': [5, 10]}
        self.clf = GridSearchCV(self.svc_model, parameters)
        GridSearchCV(estimator=self.svc_model,
                     param_grid={'C': [5, 10], 'kernel': ('linear', 'rbf')})
        self.clf.fit(self.x_train, self.y_train)
        cv_rests = self.clf.cv_results_
        self.svc_model.fit(self.x_train, self.y_train)
        self.svc_model.predict(self.x_test)
        result = pd.DataFrame({'score': cv_rests["mean_test_score"],
                               'parameters': cv_rests["params"]})
        return result

    def save_result_csv(self):
        """
             function  save_result_csv for  saving the best result into csv file
        """
        csv_result = self.s_v_m()
        parent_dir = os.getcwd()
        directory = 'Results_SVM'
        path_result = parent_dir + '/Results_SVM'
        directory_result = 'result' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        if not os.path.isdir(path_result):
            if not os.path.exists(path_result):
                os.makedirs(os.path.join(parent_dir, directory), exist_ok=True)
                result_path = os.path.join(directory, directory_result)
                csv_result = pd.DataFrame(csv_result)
                csv_result.to_csv(result_path, index=False)
        return csv_result

    def plotting_svm(self):
        """
            method for plotting  the confusion matrix
        """
        self.svc_model.predict(self.x_test)
        plot_confusion_matrix(self.clf.best_estimator_, self.x_test, self.y_test)
        path_dir = os.getcwd()
        directory = path_dir + '/Plotting_SVM'
        file = 'plot_svm'
        if not os.path.isdir(directory):
            if not os.path.exists(directory):
                os.makedirs(os.path.join(path_dir, directory), exist_ok=True)
        file_loc = os.path.join(directory, file)
        plt.savefig(file_loc + '.png')
        plt.show()
