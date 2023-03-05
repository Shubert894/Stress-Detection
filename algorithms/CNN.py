import numpy as np
import keras.losses
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import matplotlib.pyplot as plt
import sklearn.metrics as met

import algorithms.processing.primitives as pv

class MyCNN:
    def __init__(self, data, labels, sf = 512) -> None:
        self.classes = np.unique(labels)
        self.nClasses = len(self.classes)
        self.sf = sf

        self.metrics = {}
        #feature exraction
        specs = pv.get_spectrograms(data, self.sf, 2, 45)
        self.spectrograms = specs.reshape(-1, specs.shape[1], specs.shape[2],1)

        self.model = None
        self.trainHistory = None

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.spectrograms, to_categorical(labels), test_size=0.2)
        self.init_model()

    def init_model(self):
        self.model = Sequential([
            Input(shape = (self.trainX.shape[1],self.trainX.shape[2],1)),
            Conv2D(32, kernel_size =(3,3), padding = 'same'),
            LeakyReLU(alpha = 0.1),
            MaxPooling2D((2,2), padding = 'same'),
            Dropout(0.25),    

            Conv2D(64, kernel_size =(3,3), padding = 'same'),
            LeakyReLU(alpha = 0.1),
            MaxPooling2D((2,2), padding = 'same'),
            Dropout(0.4),

            Conv2D(128, kernel_size =(3,3), padding = 'same'),
            LeakyReLU(alpha = 0.1),
            MaxPooling2D((2,2), padding = 'same'),
            Dropout(0.3),
            
            Flatten(),
                    
            Dense(128),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(64),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(32),
            LeakyReLU(alpha=0.1),
            Dropout(0.3),
            Dense(self.nClasses, activation= 'softmax')])

    def train_model(self, batch_size = 16, epochs = 20):
        self.model.compile(loss= keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.trainHistory = self.model.fit(self.trainX, self.trainY, batch_size = batch_size, epochs = epochs, verbose=0, validation_data=(self.testX, self.testY))

    def evaluate(self):
        testEval = self.model.evaluate(self.testX, self.testY, verbose=0)
        print('Test loss:', testEval[0])
        print('Test accuracy:', testEval[1])
        
        accuracy = self.trainHistory.history['accuracy']
        val_accuracy = self.trainHistory.history['val_accuracy']
        loss = self.trainHistory.history['loss']
        val_loss = self.trainHistory.history['val_loss']
        epochs = range(len(accuracy))
        
        predictions = self.model.predict(self.testX)
        predictions = np.argmax(predictions, 1)
        tty = np.argmax(self.testY, 1)

        acc = met.accuracy_score(tty, predictions)
        precision = met.average_precision_score(tty, predictions)
        recall = met.recall_score(tty, predictions)
        f1 = met.f1_score(tty, predictions)
        self.metrics = {'accuracy' : acc, 'precision' : precision, 'recall' : recall, 'f1' : f1}
        self.print_metrics()

        plt.style.use('ggplot')
        fig, ax = plt.subplots(2)
        ax[0].plot(epochs, accuracy, 'r', label='Training accuracy')
        ax[0].plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        ax[0].set_title('Training and validation accuracy')
        ax[0].legend()
        ax[0].set_ylim(0,1)

        ax[1].plot(epochs, loss, 'r', label='Training loss')
        ax[1].plot(epochs, val_loss, 'b', label='Validation loss')
        ax[1].set_title('Training and validation loss')
        ax[1].legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self):
        predictions = self.model.predict(self.testX)
        predictions = np.argmax(predictions, 1)
        tty = np.argmax(self.testY, 1)
        cm = met.confusion_matrix(tty, predictions)

        import seaborn as sn
        import pandas as pd 
        df_cm = pd.DataFrame(cm, range(2), range(2))
        plt.figure(figsize=(10,7))
        sn.set(font_scale=1.6) # for label size
        sn.heatmap(df_cm, annot=True, cmap = 'copper', annot_kws={"size": 26}) # font size

        plt.show()
    
    def print_metrics(self):
        for m in self.metrics:
            print(f'{m}: {self.metrics[m]}')

if __name__ == '__main__':
    '''
        Performs CNN classification
        
        Example:
            # cnn = MyCNN(dataset['data'], dataset['labels'], 512)
            # cnn.train_model(epochs = 20)
            # cnn.evaluate()
            # cnn.plot_confusion_matrix()
    
        Explanation:    
            1)Creates an instance of the CNN class
            2)Trains the model to perform the classification
            3)Evaluates the model
            4)Plots the confusion matrix
    '''
    pass