import numpy as np
from scipy.linalg import hankel

def data_loader(params):
    data_flow = np.load('Data/X_red3.npy')
    #delay embedding
    np.random.seed(12)
    # smooth data
    window_size = 5
    kernel = np.ones(window_size) / window_size
    data_flow_smooth = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), axis=0, arr=data_flow)

    #in the current, we take all measurement and create latent variables for the unmeasurable dynamics
    data_flow_partial = data_flow_smooth
    s = ((data_flow.shape[0] - params['embedding_dimension'] + 1), (data_flow_partial.shape[1] * params[
        'embedding_dimension']))
    data_flow_H = np.zeros(s)
    for i in range(data_flow_partial.shape[1]):
        data_temp = data_flow_partial[:,i]
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

def data_loader_noH():
    data_flow = np.load('Data/X_red.npy')
    #data_flow = data_flow[:,[0,1,2,3,4]]

    data_flow = data_flow/100
    data_flow_dev = np.gradient(data_flow, 1)[0]

    train_obs = round(data_flow.shape[0]*0.8)
    x_train = data_flow[:train_obs,:]
    dx_train = data_flow_dev[:train_obs,:]
    x_val = data_flow[train_obs:, :]
    dx_val = data_flow_dev[train_obs:, :]

    return x_train, dx_train, x_val, dx_val







