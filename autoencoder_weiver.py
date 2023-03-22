import torch
import torch.nn as nn

def define_loss(network, params):
    """
    Create the loss functions.
    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']
    x_decode = network['x_decode']
    dz = network['dz']
    dz_predict = network['dz_predict']
    dx = network['dx']
    dx_decode = network['dx_decode']
    sindy_coefficients = torch.Tensor(params['coefficient_mask']).to(params['device']) * network['sindy_coefficients']

    losses = {}
    # reconstruction loss
    losses['recon'] = torch.mean((x-x_decode)**2)
    # sindy loose in z
    losses['sindy_z'] = torch.mean((dz-dz_predict)**2)
    # sindy loss in x
    losses['sindy_x'] = torch.mean((dx-dx_decode)**2)
    # sindy parameter regularization losses
    losses['sindy_regularization'] = torch.mean(torch.abs(sindy_coefficients))
    loss_total = params['loss_weight_recon']*losses['recon']\
                 +params['loss_weight_sindy_z']*losses['sindy_z']\
                 +params['loss_weight_sindy_x']*losses['sindy_x']\
                 +params['loss_weight_sindy_regularization']*losses['sindy_regularization']

    loss_refinement = params['loss_weight_recon'] * losses['recon'] \
                      + params['loss_weight_sindy_z'] * losses['sindy_z'] \
                      + params['loss_weight_sindy_x'] * losses['sindy_x']
    return loss_total, losses, loss_refinement

class linear_autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(linear_autoencoder, self).__init__()
        self.enocder = build_network_layers(input_dim, latent_dim, [], None)
        self.decoder = build_network_layers(latent_dim, input_dim, [], None)
    def forward(self, input):
        x = input
        z, encoder_weights, encoder_biases = self.enocder(x)
        x_recon, decoder_weights, decoder_biases = self.decoder(z)

        return z, x_recon, encoder_weights, encoder_biases, decoder_weights, decoder_biases

class nonlinear_autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, widths, activation = 'relu'):
        super(nonlinear_autoencoder, self).__init__()
        """
            Arguments:
                input_dim - Integer, number of state variables in the input to the first layer
                latent-dim - Integer, number of state variables in the hidden space
                widths - List of integers representing how many units are in each network layer
                activation - non-linear activation function ('relu', 'elu', 'sigmoid')
            Returns:
        """
        if activation == 'relu':
            self.activation_function = nn.ReLU()
        elif activation == 'elu':
            self.activation_function = nn.ELU()
        elif activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        else:
            raise ValueError('Invalid activation function')

        self.enocder = build_network_layers(input_dim, latent_dim, widths, self.activation_function)
        self.decoder = build_network_layers(latent_dim, input_dim, widths, self.activation_function)
    def forward(self, input):
        x = input
        z, encoder_weights, encoder_biases = self.enocder(x)
        x_recon, decoder_weights, decoder_biases = self.decoder(z)
        return z, x_recon, encoder_weights, encoder_biases, decoder_weights, decoder_biases


class build_network_layers(nn.Module):
    def __init__(self, input_dim, output_dim, widths, activation):
        super(build_network_layers, self).__init__()
        """
            Construct one portion of the network (either encoder or decoder).

            Arguments:
                input - 2D tensor array, input to the network (shape is [?,input_dim])
                input_dim - Integer, number of state variables in the input to the first layer
                output_dim - Integer, number of state variables to output from the final layer
                widths - List of integers representing how many units are in each network layer
                activation - pytorch function to be used as the activation function at each layer

            Returns:
                x - Tensor array, output of the network layers (shape is [?,output_dim])
                weights - List of torch.tensor arrays containing the network weights
                biases - List of torch.tensor arrays containing the network biases
            """
        self.activation = activation
        self.input_dim = input_dim
        self.widths = widths

        if len(self.widths) != 0:
            self.widths_input = [input_dim] + self.widths
            self.widths_input = self.widths_input[:-1]

            self.network = nn.ModuleList([nn.Linear(n_units_in, n_units, bias=True) for n_units_in, n_units in
                                          zip(self.widths_input, self.widths)])
            self.network_last = nn.Linear(self.widths[-1], output_dim, bias=True)
        else:
            self.network = []
            self.network_last = nn.Linear(self.input_dim, output_dim, bias=True)

        if self.activation is not None:
            self.activation_functions = activation

    def forward(self, input):
        weights = []
        biases = []
        x = input
        if len(self.widths) != 0:
            for i, l in enumerate(self.network):
                x = l(x)
                if self.activation is not None:
                    x = self.activation_functions(x)
                self.weights.append(l.weight.data)
                self.biases.append(l.bias.data)

            x = self.network_last(x)
            weights.append(self.network_last.weight.data)
            biases.append(self.network_last.bias.data)
        else:
            x = self.network_last(x)
            weights.append(self.network_last.weight.data)
            biases.append(self.network_last.bias.data)

        return x, weights, biases

def build_sindy_library(z, latent_dim, poly_order, include_sine = False):
    """
            Arguments:
                z - 2D pytorch tensor array of the snapshots on which to build the library. Shape is the number of time
                points by the number of state variables.
                lantent_dim - Integer, number of state variables in z.
                poly_order - Integer, polynomial order to which to build the library, max is 5
                include_sine: Boolean, whether to include sine terms in the library

            Returns:
                 tensorflow array containing the constructed library. Shape is number of time points
                 number of library functions. The number of library functions is determined by the number
                 of state variables of the input, the polynomial order, and whether sines are included.
    """
    library = [torch.ones(z.size()[0])]
    for i in range(latent_dim):
        library.append(z[:,i])
    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                library.append(z[:,i]*z[:,j])
    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])
    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])
    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])
    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, dim=1)

class z_derivative(nn.Module):
    def __init__(self):
        """
                Compute the first order time derivatives by propagating through the network.

                Arguments:
                    input - 2D pytorch array, input to the network. Dimensions are number of time points
                    by number of state variables.
                    dx - First order time derivatives of the input to the network.
                    weights - List of pytorch arrays containing the network weights
                    biases - List of pytorch arrays containing the network biases
                    activation - String specifying which activation function to use. Options are
                    'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
                    or linear.

                Returns:
                    dz - pytorch array, first order time derivatives of the network output.
            """
        super(z_derivative,self).__init__()
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input, dx, weights,biases, activation='relu'):
        dz = dx
        if activation == 'elu':
            for i in range(len(weights) - 1):
                input = torch.matmul(input, torch.transpose(weights[i],0,1)) + biases[i]
                dz = torch.mul(torch.minimum(torch.exp(input), torch.full_like(input,1.0)), torch.matmul(dz, torch.transpose(weights[i],0,1)))
                input = self.elu(input)
            dz = torch.matmul(dz, torch.transpose(weights[-1],0,1))

        elif activation == 'relu':
            for i in range(len(weights) - 1):
                input = torch.matmul(input, torch.transpose(weights[i],0,1)) + biases[i]
                dz = torch.torch.mul(torch.tensor(input > 0), torch.matmul(dz, torch.transpose(weights[i],0,1)))
                input = self.relu(input)
            dz = torch.matmul(dz, torch.transpose(weights[-1],0,1))

        elif activation == 'sigmoid':
            for i in range(len(weights) - 1):
                input = torch.matmul(input, torch.transpose(weights[i],0,1)) + biases[i]
                input = self.sigmoid(input)
                dz = torch.mul(torch.mul(input, 1 - input), torch.matmul(dz, torch.transpose(weights[i],0,1)))
            dz = torch.matmul(dz, torch.transpose(weights[-1],0,1))
        else:
            for i in range(len(weights) - 1):
                dz = torch.matmul(dz, torch.transpose(weights[i],0,1))
            dz = torch.matmul(dz, torch.transpose(weights[-1],0,1))
        return dz

class full_network(nn.Module):
    def __init__(self, params):
        super(full_network, self).__init__()
        self.params = params
        self.input_dim = params['input_dim']
        self.latent_dim = params['latent_dim']
        self.activation = params['activation']
        self.poly_order = params['poly_order']
        if 'include_sine' in params.keys():
            self.include_sine = params['include_sine']
        else:
            self.include_sine = False
        self.library_dim = params['library_dim']
        #model_order = params['model_order']
        if self.activation == 'linear':
            self.autoencoder = linear_autoencoder(self.input_dim, self.latent_dim)
        else:
            self.autoencoder = nonlinear_autoencoder(self.input_dim, self.latent_dim, self.params['widths'], activation=self.activation)
        self.z_derivative = z_derivative()

        if self.params['coefficient_initialization'] == 'xavier':
            self.sindy_coefficients = nn.Parameter(torch.rand(self.library_dim, self.latent_dim))
        elif self.params['coefficient_initialization'] == 'specified':
            self.sindy_coefficients = nn.Parameter(self.params['init_coefficients'])
        elif self.params['coefficient_initialization'] == 'constant':
            self.sindy_coefficients = nn.Parameter(torch.ones(self.library_dim, self.latent_dim))
        elif self.params['coefficient_initialization'] == 'normal':
            self.sindy_coefficients = nn.Parameter(torch.randn(self.library_dim, self.latent_dim))

        # if self.params['sequential_thresholding']:
        #     self.coefficient_mask = nn.Parameter(torch.zeros(self.library_dim, self.latent_dim))

    def forward(self, x, dx):
        network = {}
        z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = self.autoencoder(x)
        dz = self.z_derivative(x, dx, encoder_weights, encoder_biases, activation=self.activation)
        Theta = build_sindy_library(z, self.latent_dim, self.poly_order, self.include_sine)
        if self.params['sequential_thresholding']:
            coefficient_mask = torch.Tensor(self.params['coefficient_mask'])
            sindy_predict = torch.matmul(Theta, torch.mul(coefficient_mask, self.sindy_coefficients))
            network['coefficient_mask'] = coefficient_mask
        else:
            sindy_predict = torch.matmul(Theta, self.sindy_coefficients)
        dx_decode = self.z_derivative(z, sindy_predict, decoder_weights, decoder_biases, activation=self.activation)
        network['x'] = x
        network['dx'] = dx
        network['z'] = z
        network['dz'] = dz
        network['x_decode'] = x_decode
        network['dx_decode'] = dx_decode
        network['encoder_weights'] = encoder_weights
        network['encoder_biases'] = encoder_biases
        network['decoder_weights'] = decoder_weights
        network['decoder_biases'] = decoder_biases
        network['Theta'] = Theta
        network['sindy_coefficients'] = self.sindy_coefficients.data
        network['dz_predict'] = sindy_predict

        return network


