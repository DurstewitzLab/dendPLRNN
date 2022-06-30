import torch as tc
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorboardX import utils as tb_utils
import seaborn as sns

import main_eval
import utils


class Saver:
    def __init__(self, writer, save_path, args, data_set, regularizer, annealer):
        self.writer = writer
        self.save_path = save_path
        self.args = args
        self.data_set = data_set
        self.rec_model = None
        self.gen_model = None
        self.current_epoch = None
        self.current_model = None
        self.regularizer = regularizer
        self.annealer = annealer
        self.initial_save()

    def initial_save(self):
        if self.args.use_tb:
            self.save_dataset()

    def save_dataset(self):
        dataset_snippet = tc.cat(self.data_set.data)[:1000]
        plt.plot(dataset_snippet)
        plt.title('Observations')
        plt.xlabel('time steps')
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='data set', global_step=None)
        plt.close()

    def epoch_save(self, model, epoch):
        tc.save(model.rec_model.state_dict(), os.path.join(self.save_path, 'rec_model_{}.pt'.format(epoch)))
        tc.save(model.gen_model.state_dict(), os.path.join(self.save_path, 'gen_model_{}.pt'.format(epoch)))
        self.current_epoch = epoch
        self.current_model = model

        if self.args.use_tb:
            with tc.no_grad():
                self.save_loss_terms()
                # self.save_metrics()

                self.save_2d_plot()
                self.save_trial_inferred()
                self.save_trial_simulated()

                self.plot_as_image()
                self.save_parameters()

    def save_loss_terms(self):
        batch_index = 0
        batch_data = self.data_set.data[batch_index]
        if self.data_set.inputs is not None:
            batch_input = self.data_set.inputs[batch_index]
        else:
            batch_input = None
        gen_model = self.current_model.gen_model
        gen_model_parameters = gen_model.get_latent_parameters()

        z, entropy = self.current_model.rec_model.forward(batch_data)
        likelihood_x, likelihood_z = gen_model.log_likelihood(x=batch_data, z=z, s=batch_input, batch_index=batch_index)
        loss_reg = self.regularizer.loss(gen_model_parameters)

        alpha = self.annealer.get_alpha(self.current_epoch)
        loss = - (1 - alpha) * (entropy + likelihood_z) - alpha * likelihood_x + loss_reg

        A, W, h = gen_model_parameters
        AW = W + tc.diag(A)
        max_ev = tc.max(tc.abs(tc.linalg.eig(AW)[0]))
        total_norm = tc.nn.utils.clip_grad_norm_(self.current_model.get_parameters(), 1.)

        self.writer.add_scalar(tag='likelihood_x', scalar_value=likelihood_x, global_step=self.current_epoch)
        self.writer.add_scalar(tag='likelihood_z', scalar_value=likelihood_z, global_step=self.current_epoch)
        self.writer.add_scalar(tag='loss_regularization', scalar_value=loss_reg, global_step=self.current_epoch)
        self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=self.current_epoch)
        self.writer.add_scalar(tag='loss_entropy', scalar_value=entropy, global_step=self.current_epoch)
        self.writer.add_scalar(tag='set_alpha', scalar_value=alpha, global_step=self.current_epoch)
        self.writer.add_scalar(tag='max eigenvalue', scalar_value=max_ev, global_step=self.current_epoch)
        self.writer.add_scalar(tag='total_grad_norm', scalar_value=total_norm, global_step=self.current_epoch)

        print("Epoch {}/{}: Loss {:.1f}, LL_x {:.1f}, LL_z {:.1f}, H {:.1f}, max EV {:.1f}".format(
            str(self.current_epoch).zfill(4), self.args.n_epochs,
            float(loss.data), float(likelihood_x), float(likelihood_z),
            float(entropy), float(max_ev)))

    def save_metrics(self):
        """Evaluate metrics on a subset of the training data, then save them to tensorboard"""
        for metric in self.args.metrics:
            data_batch = utils.read_data(self.args.data_path)[0]
            data_subset = data_batch[:1000]
            metric_value = main_eval.eval_model_on_data_with_metric(model=self.current_model, data=data_subset, metric=metric)
            metric_value = metric_value[0]  # only take first metric value, e.g. mse 1 step ahead, and klz mc
            tag = 'metric_{}'.format(metric)
            self.writer.add_scalar(tag=tag, scalar_value=metric_value, global_step=self.current_epoch)
            print("{}: {:.1f}".format(metric, metric_value))

    def save_parameters(self):
        par_dict = {**dict(self.current_model.gen_model.state_dict()),
                    **dict(self.current_model.rec_model.state_dict())}
        par_to_tb(par_dict, epoch=self.current_epoch, writer=self.writer)

    def save_trial_inferred(self, trial_nr=0):
        trial_data = self.data_set.data[trial_nr]
        self.current_model.plot_obs_inferred(trial_data=trial_data)
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='curve trial inferred against data', global_step=self.current_epoch)
        plt.close()

    def save_trial_simulated(self, trial_nr=0):
        trial_data = self.data_set.data[trial_nr]
        time_steps = len(trial_data)
        trial_initial_state = self.current_model.get_trial_initial_state(trial_nr)

        self.current_model.plot_trial_simulated(time_steps, trial_initial_state=trial_initial_state)
        figure = plt.gcf()
        save_figure_to_tb(figure, self.writer, text='curve trial simulated', global_step=self.current_epoch)
        plt.close()
        self.current_model.plot_obs_simulated(trial_data, trial_initial_state=trial_initial_state)
        save_plot_to_tb(self.writer, text='curve trial simulated against data'.format(trial_nr),
                        global_step=self.current_epoch)
        plt.close()

    def save_2d_plot(self):
        time_steps = 10000
        x = self.current_model.gen_model.get_observed_time_series(time_steps=time_steps)
        plt.ylim(top=4, bottom=-4)
        plt.xlim(right=4, left=-4)
        plt.scatter(x[:, 0], x[:, -1], s=1)
        save_plot_to_tb(self.writer, text='curve 2d plot with transients', global_step=self.current_epoch)
        plt.close()

    def get_min_max(self, values):
        list_ = list(values)
        indices = [i for i in range(len(list_)) if list_[i] == 1]
        return min(indices), max(indices)

    def plot_trial(self, data, generated, inputs, trial_nr):
        n_units = data[0].shape[1]
        fig = plt.figure()
        for i in range(n_units):
            fig.add_subplot(n_units + 1, 1, i + 1)
            plt.plot(data[trial_nr][:, i], color='tab:blue')
            plt.plot(generated[trial_nr][:, i], color='tab:orange')

            ax = plt.gca()
            alpha = 0.1
            min_, max_ = self.get_min_max(inputs[trial_nr][:, 0])
            ax.axvspan(min_, max_, alpha=alpha, color='blue')
            min_, max_ = self.get_min_max(inputs[trial_nr][:, 1])
            ax.axvspan(min_, max_, alpha=alpha, color='green')
            min_, max_ = self.get_min_max(inputs[trial_nr][:, 2])
            ax.axvspan(min_, max_, alpha=alpha, color='red')

            plt.yticks([])

        import matplotlib.patches as mpatches
        light = mpatches.Patch(alpha=alpha, color='blue', label='cue light')
        left = mpatches.Patch(alpha=alpha, color='green', label='left lever')
        right = mpatches.Patch(alpha=alpha, color='red', label='right lever')

        import matplotlib.lines as mlines
        data_line = mlines.Line2D([], [], color='tab:blue', label='data')
        gen_line = mlines.Line2D([], [], color='tab:orange', label='generated')

        plt.legend(handles=[light, left, right, data_line, gen_line], loc='lower left')
        plt.xlabel('time steps')

    def plot_as_image(self):
        time_steps = 1000
        data_generated = self.current_model.gen_model.get_observed_time_series(time_steps=time_steps + 1000)
        data_generated = data_generated[1000:1000 + time_steps]
        data_ground_truth = self.data_set.data[0][:time_steps]
        data_generated = data_generated[:(data_ground_truth.shape[0])]  # in case trial data is shorter than time_steps

        plt.subplot(121)
        plt.title('ground truth')
        plt.imshow(data_ground_truth, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.xlabel('observations')
        plt.ylabel('time steps')
        plt.subplot(122)
        plt.title('simulated')
        plt.imshow(data_generated, aspect=0.05, origin='lower', interpolation='none', cmap='Blues_r')
        plt.ylabel('time steps')
        plt.xlabel('observations')

        save_plot_to_tb(self.writer, text='curve image'.format(), global_step=self.current_epoch)


def initial_condition_trial_to_tb(gen_model, epoch, writer):
    for i in range(len(gen_model.z0)):
        trial_z0 = gen_model.z0[i].unsqueeze(0)
        x = gen_model.get_observed_time_series(800, trial_z0)  # TODO magic length of trial
        plt.figure()
        plt.title('trial {}'.format(i))
        plt.plot(x)
        figure = plt.gcf()
        save_figure_to_tb(figure, writer, text='curve_trial{}'.format(i + 1), global_step=epoch)


def data_plot(x):
    x = x.cpu().detach().numpy()
    plt.ylim(top=4, bottom=-4)
    plt.xlim(right=4, left=-4)
    plt.scatter(x[:, 0], x[:, -1], s=3)
    plt.title('{} time steps'.format(len(x)))
    return plt.gcf()


def save_plot_to_tb(writer, text, global_step=None):
    figure = plt.gcf()
    save_figure_to_tb(figure, writer, text, global_step)


def save_figure_to_tb(figure, writer, text, global_step=None):
    image = tb_utils.figure_to_image(figure)
    writer.add_image(text, image, global_step=global_step)


def save_data_to_tb(data, writer, text, global_step=None):
    if type(data) is list:
        for i in range(len(data)):
            plt.figure()
            plt.title('trial {}'.format(i))
            plt.plot(data[i])
            figure = plt.gcf()
            save_figure_to_tb(figure=figure, writer=writer, text='curve_trial{}_data'.format(i), global_step=None)
    else:
        plt.figure()
        plt.plot(data)
        figure = plt.gcf()
        # figure = data_plot(data)
        save_figure_to_tb(figure=figure, writer=writer, text=text, global_step=global_step)


def par_to_tb(par_dict, epoch, writer):
    for key in par_dict.keys():
        par = par_dict[key]
        if len(par.shape) == 1:
            par = np.expand_dims(par, 1)
        par_to_image(par, par_name=key)
        save_plot_to_tb(writer, text='par_{}'.format(key), global_step=epoch)
        plt.close()


def par_to_image(par, par_name):
    plt.figure()
    # plt.title(par_name)
    sns.set_context('paper', font_scale=1.)
    sns.set_style('white')
    if len(par.shape)>2:
        pass
    else:
        max_dim = max(par.shape)
        use_annot = not (max_dim > 20)
        sns.heatmap(data=par, annot=use_annot, linewidths=float(use_annot), cmap='Blues_r', square=True, fmt='.2f',
                    yticklabels=False, xticklabels=False)
