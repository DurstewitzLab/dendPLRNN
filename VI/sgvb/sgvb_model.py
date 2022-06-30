import os
import torch as tc
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

import utils
from sgvb import saving
from sgvb import generative_model
from sgvb import recognition_model


def load_args(model_path):
    args_path = os.path.join(model_path, 'hypers.pkl')
    args = np.load(args_path, allow_pickle=True)
    return args


class Model(nn.Module):
    def __init__(self, args=None, data_set=None):
        super(Model, self).__init__()
        self.gen_model = None
        self.rec_model = None
        self.args = args
        self.data_set = data_set

        if args is not None:
            self.init_from_args(data_set)

    def get_parameters(self):
        return list(self.rec_model.parameters()) + list(self.gen_model.parameters())

    def init(self, rec_model, gen_model, args):
        self.rec_model = rec_model
        self.gen_model = gen_model
        self.args = args

    def init_from_args(self, data_set):
        if self.args.load_model_path is not None:
            self.init_from_model_path(self.args.load_model_path)
        else:
            self.rec_model = self.choose_and_init_rec_model()
            self.gen_model = generative_model.PLRNN(self.args.dim_x, self.args.dim_z, self.args.dim_s, self.args.n_bases, n_batches=len(data_set.data), clip_range=self.args.clip_range)

    def choose_and_init_rec_model(self):
        if self.args.rec_model == 'dc':
            rec_model = recognition_model.DiagonalCovariance(self.args.dim_x, self.args.dim_z)
        elif self.args.rec_model == 'cnn':
            rec_model = recognition_model.StackedConvolutions(self.args.dim_x, self.args.dim_z)
        else:
            raise NotImplementedError
        return rec_model

    def init_from_model_path(self, model_path, epoch=None):
        self.args = load_args(model_path)
        rec_state_dict = self.load_statedict(model_path, 'rec_model', epoch=epoch)
        self.rec_model = self.load_rec_model(rec_state_dict)
        gen_state_dict = self.load_statedict(model_path, 'gen_model', epoch=epoch)
        self.gen_model = self.load_gen_model(gen_state_dict)

    def load_rec_model(self, state_dict):
        if self.args['rec_model'] == 'dc':
            rec_model = recognition_model.DiagonalCovariance(self.args['dim_x'], self.args['dim_z'])
        elif self.args['rec_model'] == 'cnn':
            rec_model = recognition_model.StackedConvolutions(self.args['dim_x'], self.args['dim_z'])
        else:
            raise NotImplementedError
        rec_model.load_state_dict(state_dict)
        return rec_model

    def load_gen_model(self, state_dict):
        n_batches = state_dict['z0'].shape[0]
        try:
            clip_range = self.args['clip_range']
        except:
            clip_range = None
        gen_model = generative_model.PLRNN(dim_x=self.args['dim_x'], dim_z=self.args['dim_z'], dim_s=self.args['dim_s'],
                                           n_bases=self.args['n_bases'], n_batches=n_batches, clip_range=clip_range)
        gen_model.load_state_dict(state_dict)
        return gen_model

    def load_statedict(self, model_path, model_name, epoch=None):
        if epoch is None:
            epoch = self.args['n_epochs']
        path = os.path.join(model_path, '{}_{}.pt'.format(model_name, str(epoch)))
        state_dict = tc.load(path)
        return state_dict

    def eval(self):
        self.rec_model.eval()
        self.gen_model.eval()

    def train(self):
        self.rec_model.train()
        self.gen_model.train()

    def get_trial_initial_state(self, trial_nr):
        return self.gen_model.z0[trial_nr:trial_nr + 1]

    def get_trial_inferred(self, data):
        with tc.no_grad():
            Z, _ = self.rec_model(data)
            X = self.gen_model.observation(Z)
        return X.numpy(), Z.numpy()

    def get_trial_simulated(self, time_steps, trial_inputs=None, trial_initial_state=None):
        Z = self.gen_model.get_latent_time_series(time_steps=time_steps, z0=trial_initial_state, inputs=trial_inputs)
        X = self.gen_model.observation(Z)
        return X, Z

    def plot_obs_inferred(self, trial_data):
        X, Z = self.get_trial_inferred(trial_data)
        fig = plt.figure()
        plt.title('inferred observations')
        plt.axis('off')
        n_units = trial_data.shape[1]
        max_units = min([n_units, 10])
        max_time_steps = 1000
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(trial_data[:max_time_steps, i])
            plt.plot(X[:max_time_steps, i])
        plt.legend(['data', 'x_inferred'])
        plt.xlabel('time steps')

    def plot_trial_simulated(self, time_steps, trial_inputs=None, trial_initial_state=None):
        # TODO incorporate trial initial state function here
        X, Z = self.get_trial_simulated(time_steps, trial_inputs, trial_initial_state)
        fig = plt.figure()
        plt.title('simulated trial')
        plt.axis('off')
        plot_list = [X, Z]
        names = ['x', 'z']
        for i, x in enumerate(plot_list):
            fig.add_subplot(len(plot_list), 1, i + 1)
            plt.plot(x)
            plt.title(names[i])
        plt.xlabel('time steps')

    def plot_obs_simulated(self, trial_data, trial_inputs=None, trial_initial_state=None):
        time_steps = len(trial_data)
        X_simulated, Z = self.get_trial_simulated(time_steps, trial_inputs, trial_initial_state)
        fig = plt.figure()
        plt.title('observations')
        plt.axis('off')
        n_units = trial_data.shape[1]
        max_units = min([n_units, 10])
        max_time_steps = 1000
        for i in range(max_units):
            fig.add_subplot(max_units, 1, i + 1)
            plt.plot(trial_data[:max_time_steps, i])
            ax = plt.gca()
            lim = ax.get_ylim()
            plt.plot(X_simulated[:max_time_steps, i])
            ax.set_ylim(lim)
        plt.legend(['data', 'x simulated'])
        plt.xlabel('time steps')

    def plot_model_parameters(self):
        print_state_dict(self.rec_model.state_dict())
        print_state_dict(self.gen_model.state_dict())


def print_state_dict(state_dict):
    for i, par_name in enumerate(state_dict.keys()):
        par = state_dict[par_name]
        if len(par.shape) == 1:
            par = np.expand_dims(par, 1)
        plt.figure()
        axes = plt.gca()
        plt.title(par_name)
        saving.plot_par_to_axes(axes, par)


def plot_behavior(data, inputs, model):
    trial_nr = 1
    trial_data = data[trial_nr]
    trial_inputs = inputs[trial_nr]
    trial_initial_state = model.get_trial_initial_state(trial_nr)

    model.plot_obs_simulated(trial_data, trial_inputs, trial_initial_state)
    plt.show()

    data = tc.cat(data)
    inputs = tc.cat(inputs)
    model.plot_obs_simulated(data, inputs, trial_initial_state)
    plt.show()
    # model.plot_obs_inferred(trial_data)
    # plt.show()
    model.plot_trial_simulated(inputs, trial_initial_state)
    plt.show()
    # model.plot_model_parameters()
    # plt.show()


if __name__ == '__main__':
    # import sys
    # path_of_dir_above = '/'.join(sys.path[0].split('/')[:-1])
    # sys.path.append(path_of_dir_above)
    data = utils.read_data('data/lorenz96_data_noisy.npy')
    # inputs = get_data.read_npy_data('data/james_data_input.npy')
    trial_data = data[0]

    model = Model()
    model.init_from_model_path(model_path='save/test/003')
    # model.plot_obs_inferred(trial_data)
    # plt.show()
    # model.plot_obs_simulated(time_steps=1000)
    # plt.show()
    x_gen, z_gen = model.get_trial_simulated(time_steps=10000)
    for i in range(10):
        plt.subplot(1, 2, 1)
        plt.title('simulated')
        plt.imshow(x_gen[i * 1000:(i + 1) * 1000], aspect=0.05)
        plt.subplot(1, 2, 2)
        plt.title('data')
        plt.imshow(data[0][:1000], aspect=0.05)
        plt.show()
