import torch
from sindy_utils import library_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {}
params['device'] = device
params['include_SVD'] = False
params['reduced_rank'] = 17
params['latent_dim'] = 10
params['poly_order'] = 2
params['include_sine'] = False
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

## here partial measurement is the available flow station number.
params['partial_measurement'] = 6
##
params['embedding_dimension'] = 10
params['input_dim'] = params['partial_measurement'] * params['embedding_dimension']
# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.001
params['threshold_frequency'] = 1000
params['coefficient_mask'] = torch.ones((params['library_dim'], params['latent_dim'])).to(device)
params['coefficient_initialization'] = 'zeros'  # choose from 'xavier', 'specified', 'constant', 'zeros', 'normal'
# loss function weighting
params['loss_weight_recon'] = 1e-2
params['loss_weight_sindy_z'] = 1e-1
params['loss_weight_sindy_x'] = 0
params['loss_weight_sindy_regularization'] = 1e-1
params['loss_weight_z1'] = 1e-1
params['loss_weight_sindy_consistency'] = 1e-1
# training parameters
params['activation'] = 'linear'  # choose from "linear", "relu", "sigmoid", "elu"
params['widths'] = [40, 20]
params['batch_size'] = 256
params['learning_rate'] = 1e-3
# training time cutoffs
params['max_epochs'] = 3001
params['refinement_epochs'] = 1001