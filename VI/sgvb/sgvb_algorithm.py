from torch import optim
from torch import nn
import time

from sgvb import sgvb_model
from sgvb import annealing
from sgvb import regularization
from sgvb import saving


class SGVB:
    """
    Train a recognition and a generative model jointly by Stochastic Gradient Variational Bayes (SGVB).
    """

    def __init__(self, args, data_set, writer, save_path):
        self.n_epochs = args.n_epochs
        self.data_set = data_set
        self.model = sgvb_model.Model(args, data_set)
        if args.fix_observation_model:
            self.fix_observation_model()
        self.optimizer = optim.Adam(self.model.get_parameters(), args.learning_rate)
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.regularizer = regularization.Regularizer(args)
        self.annealer = annealing.Annealer(args)
        self.saver = saving.Saver(writer, save_path, args, self.data_set, self.regularizer, self.annealer)

    def fix_observation_model(self):
        for p in self.model.rec_model.parameters():
            p.requires_grad = False
        for p in self.model.gen_model.observation.parameters():
            p.requires_grad = False
        self.model.gen_model.R_x.requires_grad = False

    def loss(self, epoch, batch, batch_index):
        z, entropy = self.model.rec_model.forward(batch[0])
        likelihood_x, likelihood_z = self.model.gen_model.log_likelihood(x=batch[0], z=z, s=batch[1],
                                                                         batch_index=batch_index)
        gen_model_parameters = self.model.gen_model.get_latent_parameters()
        loss_reg = self.regularizer.loss(gen_model_parameters)

        alpha = self.annealer.get_alpha(epoch)
        loss = - (1 - alpha) * (entropy + likelihood_z) - alpha * likelihood_x + loss_reg
        return loss

    def train(self):
        self.model.train()

        for epoch in range(1, self.n_epochs + 1):
            for batch_index, batch in enumerate(self.data_set.get_dataloader()):

                self.optimizer.zero_grad()

                loss = self.loss(epoch, batch, batch_index)
                loss.backward()

                nn.utils.clip_grad_norm_(parameters=self.model.get_parameters(), max_norm=self.gradient_clipping)
                self.optimizer.step()

            if epoch % (self.n_epochs / 5) == 0:
                self.saver.epoch_save(self.model, epoch)
