import torch
from sindy_utils import library_size
import numpy as np
import os
from training_weiver import Deep_Delay_AE
from data_loader import data_loader

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('###### Preparing Data ######')
    params = {}
    params['device'] = device
    params['include_SVD'] = False
    params['reduced_rank'] = 17
    params['latent_dim'] = 10
    params['poly_order'] = 2
    params['include_sine'] = False
    params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

    ## here partial measurement is the available flow station number.
    params['partial_measurement'] = 7
    ##
    params['embedding_dimension'] = 5
    params['input_dim'] = params['partial_measurement'] * params['embedding_dimension']
    # sequential thresholding parameters
    params['sequential_thresholding'] = True
    params['coefficient_threshold'] = 0.1
    params['threshold_frequency'] = 500
    params['coefficient_mask'] = torch.ones((params['library_dim'], params['latent_dim'])).to(device)
    params['coefficient_initialization'] = 'constant'
    # loss function weighting
    params['loss_weight_recon'] = 1e-2
    params['loss_weight_sindy_z'] = 1e-1
    params['loss_weight_sindy_x'] = 0
    params['loss_weight_sindy_regularization'] = 1e-1
    params['loss_weight_z1'] = 1e-1
    params['loss_weight_sindy_consistency'] = 1e-20
    # training parameters
    params['activation'] = 'linear'  # choose from "linear", "relu", "sigmoid", "elu"
    params['widths'] = [14, 10]
    params['batch_size'] = 256
    params['learning_rate'] = 1e-3
    # params['data_path'] = os.getcwd() + '/'
    # training time cutoffs
    params['max_epochs'] = 5001
    params['refinement_epochs'] = 1001

    x_train, dx_train, x_valid, dx_valid = data_loader(params)

    model = Deep_Delay_AE(params).to(device)
    score, losses, train_loss_epochs, val_loss_epochs, refine_loss_epochs, refine_val_loss_epochs = model.Train(x_train, dx_train, x_valid, dx_valid)

    import pickle

    with open('Results/final_output.pkl', 'wb') as f:
        pickle.dump(score, f)
    with open('Results/final_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)


    train_loss_epochs = [i.cpu().detach().numpy() for i in train_loss_epochs]
    val_loss_epochs = [i.cpu().detach().numpy() for i in val_loss_epochs]
    refine_loss_epochs = [i.cpu().detach().numpy() for i in refine_loss_epochs]
    refine_val_loss_epochs = [i.cpu().detach().numpy() for i in refine_val_loss_epochs]

    np.save('Results/train_loss.npy', train_loss_epochs)
    np.save('Results/val_loss.npy', val_loss_epochs)
    np.save('Results/refine_loss.npy', refine_loss_epochs)
    np.save('Results/refine_val_loss.npy', refine_val_loss_epochs)


