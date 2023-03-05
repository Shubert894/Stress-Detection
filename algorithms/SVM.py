from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import sklearn.metrics as met
import matplotlib.pyplot as plt

import algorithms.processing.primitives as pv

class MySVM:  
    def __init__(self, data, labels, sf) -> None:
        spectrograms = pv.get_spectrograms(data, sf, 2, 45)
        spectrograms = spectrograms.reshape(-1, spectrograms.shape[1]*spectrograms.shape[2])
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(spectrograms, labels, test_size=0.2)
        self.typeSVM = None
        self.model = None
        self.metrics = {}

    def classify(self):
        if len(self.trainX) <= 5:
            print('Not enough data')
            return
        
        self.typeSVM = 'c'
        self.model = SVC()

        param_grid = {'kernel': ['rbf', 'linear'], 'C': [1.0, 10.0]}
        grid_search = GridSearchCV(self.model, param_grid, cv = 4, refit=True)

        grid_search.fit(self.trainX, self.trainY)

        best_params = grid_search.best_params_
        print(best_params)

        self.model = grid_search.best_estimator_

        predictions = self.model.predict(self.testX)
        accuracy = met.accuracy_score(self.testY, predictions)
        precision = met.average_precision_score(self.testY, predictions)
        recall = met.recall_score(self.testY, predictions)
        f1 = met.f1_score(self.testY, predictions)
        self.metrics = {'accuracy' : accuracy, 'precision' : precision, 'recall' : recall, 'f1' : f1}
        self.print_metrics()

    def regress(self):
        if len(self.trainX) <= 5:
            print('Not enough data')
            return
        
        self.typeSVM = 'r'
        self.model = SVR()
        
        param_grid = {'kernel': ['rbf', 'linear'], 'C': [1.0, 10.0], 'epsilon': [0.1, 0.2, 0.3]}
        grid_search = GridSearchCV(self.model, param_grid, cv = 4, refit=True)

        grid_search.fit(self.trainX, self.trainY)

        best_params = grid_search.best_params_
        print(best_params)

        self.model = grid_search.best_estimator_

        predictions = self.model.predict(self.testX)
        mse = mean_squared_error(self.testY, predictions)
        r2 = r2_score(self.testY, predictions)
        self.metrics = {'r2' : r2, 'mse' : mse}
        self.print_metrics()
    
    def print_metrics(self):
        for m in self.metrics:
            print(f'{m}: {self.metrics[m]}')
    
    def plot_confusion_matrix(self):
        predictions = self.model.predict(self.testX)
        cm = met.confusion_matrix(self.testY, predictions)

        import seaborn as sn
        import pandas as pd 
        df_cm = pd.DataFrame(cm, range(2), range(2))
        plt.figure(figsize=(10,7))
        sn.set(font_scale=1.6) # for label size
        sn.heatmap(df_cm, annot=True, cmap = 'copper', annot_kws={"size": 26}) # font size

        plt.show()



if __name__ == '__main__':
    '''
        Performs SVM classification or regression
        
        Example:
            svm = MySVM(data, labels, 512)
            svm.classify()
            svm.plot_confusion_matrix()
    
        Explanation:    
            1)Creates an instance of the SVM
            2)Trains the model to perform the classification
            3)Plots the confusion matrix
    '''
    pass