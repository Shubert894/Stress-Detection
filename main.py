import os
import numpy as np
from pathlib import Path

import vis as vv
import algorithms.processing.primitives as pv
from algorithms.SVM import MySVM
from algorithms.CNN import MyCNN

def main():
    '''
        Main function that gets executed.
    '''

    path = os.path.join(Path.cwd(), Path("data/conventioned/stress_exp_1.json"))
    dataset = pv.get_data(path)
    data = np.array(dataset['data'])
    labels = np.array(dataset['labels'])

    #---Inspection of the 44th fragment of the signal and the spectrograms of the first 15 fragments---
    
    # vv.preview(data[44],512)
    # vv.view_spectrograms(data)

    
    #---SVM classification---
    
    # svm = MySVM(data, labels, 512)
    # svm.classify()
    # svm.plot_confusion_matrix()

    
    #---CNN classification---
    
    # cnn = MyCNN(dataset['data'], dataset['labels'], 512)
    # cnn.train_model(epochs = 20)
    # cnn.evaluate()
    # cnn.plot_confusion_matrix()
    
    
    #---PCA visualization of data---
    
    # vv.pca(data, labels, 512, 3)


if __name__ == '__main__':
    main()
