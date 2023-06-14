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
    score, losses, train_loss_epochs, val_loss_epochs, refine_loss_epochs, refine_val_loss_epochs, \
        train_loss_recon_epochs, val_loss_recon_epochs, train_loss_x_epochs, val_loss_x_epochs,\
        train_loss_z_epochs, val_loss_z_epochs, train_loss_z1_epochs, val_loss_z1_epochs, \
        train_loss_cons_epochs, val_loss_cons_epochs, train_loss_reg_epochs, val_loss_reg_epochs = model.Train(x_train, dx_train, x_valid, dx_valid)

    import pickle

    with open('Results/final_output.pkl', 'wb') as f:
        pickle.dump(score, f)
    with open('Results/final_losses.pkl', 'wb') as f:
        pickle.dump(losses, f)


    train_loss_epochs = [i.cpu().detach().numpy() for i in train_loss_epochs]
    val_loss_epochs = [i.cpu().detach().numpy() for i in val_loss_epochs]
    refine_loss_epochs = [i.cpu().detach().numpy() for i in refine_loss_epochs]
    refine_val_loss_epochs = [i.cpu().detach().numpy() for i in refine_val_loss_epochs]

    train_loss_recon_epochs = [i.cpu().detach().numpy() for i in train_loss_recon_epochs]
    val_loss_recon_epochs = [i.cpu().detach().numpy() for i in val_loss_recon_epochs]

    train_loss_z_epochs = [i.cpu().detach().numpy() for i in train_loss_z_epochs]
    val_loss_z_epochs = [i.cpu().detach().numpy() for i in val_loss_z_epochs]

    train_loss_x_epochs = [i.cpu().detach().numpy() for i in train_loss_x_epochs]
    val_loss_x_epochs = [i.cpu().detach().numpy() for i in val_loss_x_epochs]

    train_loss_z1_epochs = [i.cpu().detach().numpy() for i in train_loss_z1_epochs]
    val_loss_z1_epochs = [i.cpu().detach().numpy() for i in val_loss_z1_epochs]

    train_loss_cons_epochs = [i.cpu().detach().numpy() for i in train_loss_cons_epochs]
    val_loss_cons_epochs = [i.cpu().detach().numpy() for i in val_loss_cons_epochs]

    train_loss_reg_epochs = [i.cpu().detach().numpy() for i in train_loss_reg_epochs]
    val_loss_reg_epochs = [i.cpu().detach().numpy() for i in val_loss_reg_epochs]

    np.save('Results/train_loss.npy', train_loss_epochs)
    np.save('Results/val_loss.npy', val_loss_epochs)
    np.save('Results/refine_loss.npy', refine_loss_epochs)
    np.save('Results/refine_val_loss.npy', refine_val_loss_epochs)
    np.save('Results/train_loss_recon.npy', train_loss_recon_epochs)
    np.save('Results/val_loss_recon.npy', val_loss_recon_epochs)
    np.save('Results/train_loss_x.npy', train_loss_x_epochs)
    np.save('Results/val_loss_x.npy', val_loss_x_epochs)
    np.save('Results/train_loss_z.npy', train_loss_z_epochs)
    np.save('Results/val_loss_z.npy', val_loss_z_epochs)
    np.save('Results/train_loss_z1.npy', train_loss_z1_epochs)
    np.save('Results/val_loss_z1.npy', val_loss_z1_epochs)
    np.save('Results/train_loss_cons.npy', train_loss_cons_epochs)
    np.save('Results/val_loss_cons.npy', val_loss_cons_epochs)
    np.save('Results/train_loss_reg.npy', train_loss_reg_epochs)
    np.save('Results/val_loss_reg.npy', val_loss_reg_epochs)




