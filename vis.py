import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import algorithms.processing.primitives as pv

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams['axes.titlesize'] = 20


def view_spectrograms(data, sf = 512, minF = 2, maxF = 45):
    '''
        Displays multiple spectrograms given data
    '''
    spectrograms = pv.get_spectrograms(data,sf,minF,maxF)
    n = len(data)
    
    fig, axs = plt.subplots(3,5)
    
    for i in range(3):
        for j in range(5):
            axs[i][j].imshow(spectrograms[i*5+j], aspect='auto', extent= [0, 2, 45, 0])
            
            if i != 2:
                axs[i,j].get_xaxis().set_visible(False)
            if j != 0:
                axs[i,j].get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # img = ax.imshow(spectrograms[0])
    # axcolor = 'yellow'
    # ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
    # slider = Slider(ax_slider, '', 0, n-1, valinit=0, valstep=1)
    # def update(val):
    #     ax.imshow(spectrograms[int(val)])
    #     fig.canvas.draw_idle()
    # slider.on_changed(update)
    # plt.show()


def pca(data, labels, sf, dimensions = 2):
    '''
        Performs Principal Component Analysis on the data and colors the data points acording to their label.
        
        Possible number of PCA dimensions: 2D, 3D
    '''

    spectrograms = pv.get_spectrograms(data, sf, 2, 45)
    spectrograms = spectrograms.reshape(-1, spectrograms.shape[1]*spectrograms.shape[2])
    
    pca = PCA(n_components=dimensions)
    pca_features = pca.fit_transform(data)
    print(f'Explained variance PCA {dimensions}D: {pca.explained_variance_}')
    
    if dimensions == 2:
        plt.scatter(pca_features[:,0],pca_features[:,1], c = labels)
        plt.show()
    else:
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pca_features[:,0],pca_features[:,1],pca_features[:,2], c=labels, marker='o')
        plt.show()


def show_spectrogram(sig, sf, minF, maxF):
    '''
        Shows the spectrogram of the signal
    '''
    f, t, Sxx = pv.get_spectrogram(sig, sf, minF, maxF)
    plt.pcolormesh(t, f, Sxx)
    plt.colorbar()
    plt.show()


def preview(sig, sf):
    '''
        Function to inspect a signal. It plots the standardized data, together with the
        power specrum of it.
    '''
    
    timeScale = np.arange(0,len(sig)/sf,1/sf)
    nSig = pv.standardize(sig)
    fsc, power = pv.get_power(nSig, sf)
    

    fig, axs = plt.subplots(2)
    fig.subplots_adjust(hspace=0.5)
    axs[0].set_title('Standardized Signal')
    axs[0].set_xlim((0,timeScale[-1]))
    axs[0].plot(timeScale, nSig)
    axs[1].set_title('Power Spectrum')
    axs[1].set_xlim((0,fsc[-1]))
    axs[1].plot(fsc, power)
    plt.show()    

if __name__ == '__main__':
    pass