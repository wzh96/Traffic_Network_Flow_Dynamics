import numpy as np
from scipy.linalg import hankel

def data_loader(params):
    data_flow = np.load('Data/X.npy')
    #delay embedding
    #partial_dim =
    partial_dim = params['partial_measurement']
    np.random.seed(12)
    partial_index = np.sort(np.random.choice(data_flow.shape[1], partial_dim,replace=False))
    data_flow_partial = data_flow[:,partial_index]
    s = ((data_flow.shape[0] - params['embedding_dimension'] + 1), (data_flow_partial.shape[1] * params[
        'embedding_dimension']))
    data_flow_H = np.zeros(s)
    for i in range(data_flow_partial.shape[1]):
        data_temp = data_flow_partial[:,0]
        data_temp_H = hankel(data_temp)
        data_temp_H = data_temp_H[:(data_flow.shape[0]-params['embedding_dimension']+1), :params['embedding_dimension']]
        data_flow_H[:,i*params['embedding_dimension']:(i+1)*params['embedding_dimension']] = data_temp_H
    data_flow_H = data_flow_H/100
    data_flow_H_dev = np.gradient(data_flow_H, 1)[0]

    train_obs = round(data_flow_H.shape[0]*0.8)
    x_train = data_flow_H[:train_obs,:]
    dx_train = data_flow_H_dev[:train_obs,:]
    x_val = data_flow_H[train_obs:, :]
    dx_val = data_flow_H_dev[train_obs:, :]

    return x_train, dx_train, x_val, dx_val




