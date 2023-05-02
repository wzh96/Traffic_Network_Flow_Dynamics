import torch
import numpy as np
from training_weiver import Deep_Delay_AE
from data_loader import data_loader
from utils import params

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('###### Preparing Data ######')

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


