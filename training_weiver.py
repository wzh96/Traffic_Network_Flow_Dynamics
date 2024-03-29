import torch
import torch.nn as nn
import time
import numpy as np
from Deep_Delay_AutoEncoder_Wei import full_network
from Deep_Delay_AutoEncoder_Wei import define_loss

class Deep_Delay_AE(nn.Module):
    def __init__(self, params):
        super(Deep_Delay_AE, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.network = full_network(self.params).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(),lr=self.params['learning_rate'])
        self.LRSchdular = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma= 0.5)
        #self.optimizer_refine = torch.optim.Adam(self.network.parameters(),lr=self.params['learning_rate'])

    def Train(self, x_train, dx_train, x_val, dx_val):
        self.network.train()
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.params['batch_size']
        print("###### model training in process ######")
        x_train, dx_train = torch.Tensor(x_train).to(self.device), torch.Tensor(dx_train).to(self.device)
        x_val, dx_val = torch.Tensor(x_val).to(self.device), torch.Tensor(dx_val).to(self.device)

        training_loss_sum = []
        training_loss_val_sum = []
        training_loss_recon_sum = []
        training_loss_val_recon_sum = []
        training_loss_x_sum = []
        training_loss_val_x_sum = []
        training_loss_z_sum = []
        training_loss_val_z_sum = []
        training_loss_z1_sum = []
        training_loss_val_z1_sum = []
        training_loss_cons_sum = []
        training_loss_val_cons_sum = []
        training_loss_reg_sum = []
        training_loss_val_reg_sum = []

        for epoch in range(self.params['max_epochs']):
            start_time = time.time()
            for b in range(num_batches):
                x_train_batch = x_train[b * self.params['batch_size']:(b+1)*self.params['batch_size'],:]
                dx_train_batch = dx_train[b * self.params['batch_size']:(b+1)*self.params['batch_size'],:]
                score = self.network(x_train_batch, dx_train_batch)
                loss,losses_all,_ = define_loss(score, self.params)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print('Batch {:d}/{:d} Loss {:.6f}'.format(b, num_batches, loss), end='\r', flush=True)
            if self.params['sequential_thresholding'] and (epoch % self.params['threshold_frequency'] == 0) and (epoch > 0):
                self.params['coefficient_mask'] = torch.abs(score['sindy_coefficients']) > self.params['coefficient_threshold']
                print('THRESHOLDING: %d active coefficients' % torch.sum(self.params['coefficient_mask']))
            self.LRSchdular.step()
            duration = time.time() - start_time
            score_val = self.network(x_val, dx_val)
            loss_val,losses_val_all,_ = define_loss(score_val, self.params)
            print('Epoch {:d} Loss {:.6f} Validation Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, loss_val, duration))

            training_loss_sum.append(loss)
            training_loss_val_sum.append(loss_val)

            training_loss_recon_sum.append(losses_all['recon'])
            training_loss_val_recon_sum.append(losses_val_all['recon'])

            training_loss_x_sum.append(losses_all['sindy_x'])
            training_loss_val_x_sum.append(losses_val_all['sindy_x'])

            training_loss_z_sum.append(losses_all['sindy_z'])
            training_loss_val_z_sum.append(losses_val_all['sindy_z'])

            training_loss_z1_sum.append(losses_all['z1_loss'])
            training_loss_val_z1_sum.append(losses_val_all['z1_loss'])

            training_loss_cons_sum.append(losses_all['sindy_consistency_loss'])
            training_loss_val_cons_sum.append(losses_val_all['sindy_consistency_loss'])

            training_loss_reg_sum.append(losses_all['sindy_regularization'])
            training_loss_val_reg_sum.append(losses_val_all['sindy_regularization'])

        print('###### Model refinement in process ######')
        refine_loss_sum = []
        refine_loss_val_sum = []
        for epoch_refine in range(self.params['refinement_epochs']):
            start_time = time.time()
            for b in range(num_batches):
                x_train_batch = x_train[b * self.params['batch_size']:(b + 1) * self.params['batch_size'], :]
                dx_train_batch = dx_train[b * self.params['batch_size']:(b + 1) * self.params['batch_size'], :]
                x_train_batch, dx_train_batch = torch.Tensor(x_train_batch).to(self.device), torch.Tensor(
                    dx_train_batch).to(self.device)
                score = self.network(x_train_batch, dx_train_batch)
                _, _, loss_refine = define_loss(score, self.params)
                self.optimizer.zero_grad()
                loss_refine.backward()
                self.optimizer.step()
                print('Batch {:d}/{:d} Loss_Refinement {:.6f}'.format(b, num_batches, loss_refine), end='\r', flush=True)
            self.LRSchdular.step()
            duration = time.time() - start_time
            score_val = self.network(x_val, dx_val)
            _,_,loss_refine_val = define_loss(score_val, self.params)
            print('Epoch {:d} Refinement loss {:.6f} Validation refinement loss {:.6f} Duration {:.3f} seconds.'.format(epoch_refine, loss_refine, loss_refine_val, duration))
            refine_loss_sum.append(loss_refine)
            refine_loss_val_sum.append(loss_refine_val)
        # output all losses
        _,losses_detail_final,_ = define_loss(score_val, self.params)

        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, 'Saved_Model/model_checkpoint.pt')


        # Save the validation score
        results_dict = {'sindy_coefficients': torch.mul(self.params['coefficient_mask'], score['sindy_coefficients']),
                        'x': score_val['x'],
                        'dx': score_val['dx'],
                        'z': score_val['z'],
                        'dz': score_val['dz'],
                        'x_decode': score_val['x_decode'],
                        'dx_decode': score_val['dx_decode'],
                        'dz_predict': score_val['dz_predict'],
                        'Theta': score_val['Theta'],
                        # 'encoder_weights': score_val['encoder_weights'],
                        # 'encoder_biases': score_val['encoder_biases'],
                        # 'decoder_weights': score_val['decoder_weights'],
                        # 'decoder_biases': score_val['decoder_biases']
                        }

        # move all items in the dict to 'cpu'

        for key in results_dict:
            if isinstance(results_dict[key], torch.Tensor):
                results_dict[key] = results_dict[key].cpu()

        for key in losses_detail_final:
            if isinstance(losses_detail_final[key], torch.Tensor):
                losses_detail_final[key] = losses_detail_final[key].cpu()

        return results_dict, losses_detail_final, training_loss_sum, training_loss_val_sum, refine_loss_sum, refine_loss_val_sum, training_loss_recon_sum, \
               training_loss_val_recon_sum, training_loss_x_sum, training_loss_val_x_sum, training_loss_z_sum, training_loss_val_z_sum, training_loss_z1_sum, \
               training_loss_val_z1_sum, training_loss_cons_sum, training_loss_val_cons_sum, training_loss_reg_sum, training_loss_val_reg_sum





