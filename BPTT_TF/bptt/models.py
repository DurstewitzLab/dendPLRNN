import os
import torch as tc
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import utils
from bptt import saving
from bptt import PLRNN_model
import random
from typing import Optional


def load_args(model_path):
    args_path = os.path.join(model_path, 'hypers.pkl')
    args = np.load(args_path, allow_pickle=True)
    return args

class Model(nn.Module):
    def __init__(self, args=None, data_set=None):
        super().__init__()
        self.latent_model = None
        self.device = None
        self.args = args
        self.data_set = data_set

        if args is not None:
            # cast args to dictionary
            self.args = vars(args)
            self.init_from_args()

    def forward(self, x, tau=None, z0=None):
        B = None
        # teacher forcing with obs. model inversion
        if self.args['use_inv_tf']:
            B = self.output_layer.weight

        if self.z0_model:
            z0 = self.z0_model(x[:, 0, :])

        hidden_out = self.latent_model(x, tau, z0, B)
        output = self.output_layer(hidden_out)
        return output

    def to(self, device: tc.device):
        self.device = device
        return super().to(device)

    def get_latent_parameters(self):
        '''
        Return a list of all latent model parameters:
        A: (dz, )
        W: (dz, dz)
        h: (dz, )

        For BE-models additionally:
        alpha (db, )
        thetas (dz, db)
        '''
        return self.latent_model.get_parameters()

    def get_num_trainable(self):
        '''
        Return the number of total trainable parameters
        '''
        return sum([p.numel() if p.requires_grad else 0 
                    for p in self.parameters()])

    def init_from_args(self):
        # resume from checkpoint?
        model_path = self.args['load_model_path']
        if model_path is not None:
            epoch = None
            if self.args['resume_epoch'] is None:
                epoch = utils.infer_latest_epoch(model_path)
            self.init_from_model_path(model_path, epoch)
        else:
            self.init_submodules()

    def init_z0_model(self, learn_z0: bool):
        z0model = None
        if learn_z0 and not self.args['use_inv_tf']:
            z0model = Z0Model(self.args['dim_x'], self.args['dim_z'])
        return z0model

    def init_obs_model(self, fix_output_layer):
        dz, dx = self.args['dim_z'], self.args['dim_x']
        output_layer = None
        if fix_output_layer:
            # fix observation model (identity mapping z<->x)
            output_layer = nn.Linear(dz, dx, bias=False)
            B = tc.zeros((dx, dz))
            for i in range(dx):
                B[i, i] = 1
            output_layer.weight = nn.Parameter(B, False)
        else:
            # learnable 
            output_layer = nn.Linear(dz, dx, bias=False)
        return output_layer

    def init_from_model_path(self, model_path, epoch=None):
        # load arguments
        self.args = load_args(model_path)

        # init using arguments
        self.init_submodules()

        # restore model parameters
        self.load_state_dict(self.load_statedict(model_path, 'model', epoch=epoch))

    def init_submodules(self):
        '''
        Initialize latent model, output layer and z0 model.
        '''
        # TODO: Add RNN/LSTM as separate models w/o explicit latent model..
        self.latent_model = PLRNN_model.PLRNN(self.args['dim_x'], self.args['dim_z'], 
                                              self.args['n_bases'], 
                                              latent_model=self.args['latent_model'], 
                                              clip_range=self.args['clip_range'],
                                              mean_centering=self.args['mean_centering'],
                                              dataset=self.data_set)

        self.output_layer = self.init_obs_model(self.args['fix_obs_model'])
        self.z0_model = self.init_z0_model(self.args['learn_z0'])

    def load_statedict(self, model_path, model_name, epoch=None):
        if epoch is None:
            epoch = self.args['n_epochs']
        path = os.path.join(model_path, '{}_{}.pt'.format(model_name, str(epoch)))
        state_dict = tc.load(path)
        return state_dict

    @tc.no_grad()
    def generate_free_trajectory(self, data, T, z0=None):
        B = None
        # teacher forcing with obs. model inversion
        if self.args['use_inv_tf']:
            B = self.output_layer.weight
        # optionally predict an initial z0 of shape (1, dz)
        if self.z0_model:
            z0 = self.z0_model(data[0, :].unsqueeze(0))
        # latent traj is T x dz
        latent_traj = self.latent_model.generate(T, data, z0, B)
        # cast to b x T x dz for output layer and back to T x dx
        obs_traj = self.output_layer(latent_traj.unsqueeze(0)).squeeze_(0)
        # T x dx, T x dz
        return obs_traj, latent_traj

    def plot_simulated(self, data: tc.Tensor, T: int):
        X, Z = self.generate_free_trajectory(data, T)
        fig = plt.figure()
        plt.title('simulated')
        plt.axis('off')
        plot_list = [X, Z]
        names = ['x', 'z']
        for i, x in enumerate(plot_list):
            fig.add_subplot(len(plot_list), 1, i + 1)
            plt.plot(x.cpu())
            plt.title(names[i])
        plt.xlabel('time steps')

    def plot_obs_simulated(self, data: tc.Tensor):
        time_steps = len(data)
        X, Z = self.generate_free_trajectory(data, time_steps)
        fig = plt.figure()
        plt.title('observations')
        plt.axis('off')
        n_units = data.shape[1]
        max_units = min([n_units, 10])
        max_time_steps = 1000
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(data[:max_time_steps, i].cpu())
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(X[:max_time_steps, i].cpu())
            ax.set_ylim(lim)
        plt.legend(['data', 'x simulated'])
        plt.xlabel('time steps')

    @tc.no_grad()
    def plot_prediction(self, data: tc.Tensor, 
                        rand_seq: Optional[bool] = True):
        '''
        Plot prediction of the model for a given
        input sequence with teacher forcing (interleaved
        observations)
        '''
        T = self.args['seq_len']
        N = self.args['teacher_forcing_interval']
        T_full, dx = data.size()
        max_units = min([dx, 10])

        # input and model prediction
        if rand_seq:
            t = random.randint(0, T_full - T)
        else:
            t = 0
        input_ = data[t : t + T]
        pred = self(input_.unsqueeze(0), N).squeeze(0)

        # x axis
        x = np.arange(T-1)

        # plot
        fig = plt.figure()
        plt.title('Prediction')
        plt.axis('off')
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(x, input_[1:, i].cpu(), label='GT')
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(x, pred[:-1, i].cpu(), label='Pred')
            ax.set_ylim(lim)
            plt.scatter(x[::N], input_[1::N, i].cpu(), marker='2', 
                        label='TF-obs', color='r')
            ax.set_ylim(lim)
            plt.legend(prop={"size": 5})
            plt.ylabel(f'$x_{i}$')
        plt.xlabel('t')


class Z0Model(nn.Module):
    '''
    MLP that predicts an optimal initial latent state z0 given
    an inital observation x0.

    Takes x0 of dimension dx and returns z0 of dimension dz, by
    predicting dz-dx states and then concatenating x0 and the prediction:
    z0 = [x0, MLP(x0)]
    '''
    def __init__(self, dx: int, dz: int):
        super(Z0Model, self).__init__()
        # TODO: MLP currently only affine transformation
        # maybe try non-linear, deep variants?
        self.MLP = nn.Linear(dx, dz-dx, bias=False)

    def forward(self, x0):
        return tc.cat([x0, self.MLP(x0)], dim=1)
