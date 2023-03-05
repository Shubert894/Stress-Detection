import json
import numpy as np
import scipy.signal as ss


def standardize(a):
    mean = np.mean(a)
    std = np.std(a)
    return (a-mean)/std

def normalize(a):
    diff = a.max() - a.min()
    if diff == 0:
        return a
    else:
        return (a - a.min())/diff

def get_power(a, sf, minF = 2, maxF = 45):
    freqScale, power = ss.periodogram(a, sf, window='tukey', scaling='density')
    argMaxF = np.argmin(np.abs(freqScale-maxF))
    argMinF = np.argmin(np.abs(freqScale-minF))
    freqScale = freqScale[argMinF:argMaxF]
    power = standardize(power[argMinF:argMaxF])
    return freqScale, power

def get_data(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)
    
def dump_data(file_name, data):
    with open(file_name, 'w') as f:
        return json.dump(data, f)


def filter_data(sig, sf = 512):
    sig = np.array(sig)
    sig[np.abs(sig)>1500] = 0
    n = 100
    Wn = [1/(sf//2),45/(sf//2)]
    b = ss.firwin(n, Wn, pass_zero='bandpass')
    return ss.filtfilt(b, 1, sig)

def get_spectrogram(a, sf, minF = 2, maxF = 45):
    f, t, Sxx = ss.spectrogram(standardize(a), sf, nperseg = sf//2 , noverlap = sf//4)
    argMaxF = np.argmin(np.abs(f-maxF))
    argMinF = np.argmin(np.abs(f-minF))
    f = f[argMinF:argMaxF]
    Sxx = Sxx[argMinF:argMaxF,:]
    Sxx = normalize(Sxx)
    return f, t, Sxx

def get_spectrograms(data, sf, minF = 2, maxF = 45):
    spectrograms = []
    for s in data:
        f, t, Sxx = get_spectrogram(s, sf, minF, maxF)
        Sxx = normalize(Sxx)
        spectrograms.append(Sxx)
    return np.array(spectrograms)


if __name__ == '__main__':
    pass