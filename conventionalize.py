import numpy as np
import os
from pathlib import Path

import algorithms.processing.primitives as pv

def convert():
    from_path = os.path.join(Path.cwd(), Path("data/unconventioned/stress_exp_1.json"))
    to_path = os.path.join(Path.cwd(), Path("data/conventioned/stress_exp_1.json"))
    
    dataset = pv.get_data(from_path)
    dict = {}
    dict['name'] = 'Stress_exp_1'
    dict['pdc'] = 'pdc1'
    dict['sampling'] = 512

    data = []
    labels = []

    contr = dataset["1"]
    stressed = dataset["2"]

    for i in range(len(contr)//1024):
        snip = contr[1024*i:1024*(i+1)]
        snip = np.array(snip)
        snip[np.abs(snip)>2000] = 0
        snip = pv.filter_data(snip)
        snip = pv.standardize(snip)
        snip = snip.tolist()
        data.append(snip)
        labels.append(0)
    
    for i in range(len(stressed)//1024):
        snip = stressed[1024*i:1024*(i+1)]
        snip = np.array(snip)
        snip[np.abs(snip)>2000] = 0
        snip = pv.filter_data(snip)
        snip = pv.standardize(snip)
        snip = snip.tolist()
        data.append(snip)
        labels.append(1)
    
    dict['data'] = data
    dict['labels'] = labels

    pv.dump_data(to_path, dict)


if __name__ == '__main__':
    '''
        The following script is supposed to transform data from an unconventioned format to one supported
        by a convention specified in 'data_conventions.txt'.

        Not all data formats are supported
    '''
    convert()