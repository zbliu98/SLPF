import os
import numpy as np
import pandas as pd 
def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMS04':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'BAY':
        data_path = os.path.join('../data/BAY/pems-bay.h5')
        data = pd.read_hdf(data_path).to_numpy()
    elif dataset == 'LA':
        data_path = os.path.join('../data/LA/metr-la.h5')
        data = pd.read_hdf(data_path).to_numpy()  # onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load {} Dataset shaped: {}, Max: {}, Min: {}, Mean: {}, Median: {}'.format(dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data)))
    return data
