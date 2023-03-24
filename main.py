import torch
from sindy_utils import library_size
import numpy as np
import os
from training_weiver import Deep_Delay_AE
from examples.lorenz.example_lorenz import get_lorenz_data
from data_loader import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('###### Preparing Data ######')

params = {}
params['device'] = device
params['include_SVD'] = False
params['reduced_rank'] = 17
params['latent_dim'] = 7
params['poly_order'] = 2
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)
params['partial_measurement'] = 6
params['embedding_dimension'] = 3
params['input_dim'] = params['partial_measurement'] * params['embedding_dimension']
# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.01
params['threshold_frequency'] = 500
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'
# loss function weighting
params['loss_weight_recon'] = 1e-2
params['loss_weight_sindy_z'] = 1e-2
params['loss_weight_sindy_x'] = 1e-2
params['loss_weight_sindy_regularization'] = 1e-2
params['loss_weight_z1'] = 1e-1
params['loss_weight_sindy_consistency'] = 1e-1
#params['loss_weight_negative_z'] = 100
# training parameters
params['activation'] = 'linear' # choose from "linear", "relu", "sigmoid", "elu"
params['widths'] = [14,10]
params['batch_size'] = 256
params['learning_rate'] = 1e-3
#params['data_path'] = os.getcwd() + '/'
# training time cutoffs
params['max_epochs'] = 10001
params['refinement_epochs'] = 1001

x_train, dx_train, x_valid, dx_valid = data_loader(params)

model = Deep_Delay_AE(params).to(device)
score, losses = model.Train(x_train,dx_train, x_valid, dx_valid)

import pickle
with open('Results/final_output.pkl', 'wb') as f:
    pickle.dump(score, f)
with open('Results/final_losses.pkl', 'wb') as f:
    pickle.dump(losses, f)
