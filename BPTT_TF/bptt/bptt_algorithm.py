from copy import deepcopy
import torch as tc
from torch import optim
from torch import nn
from bptt import models
from bptt import regularization
from bptt import saving
from bptt.dataset import GeneralDataset
from tensorboardX import SummaryWriter
from argparse import Namespace
from timeit import default_timer as timer
import datetime
from bptt.tau_estimation import estimate_forcing_interval

class BPTT:
    def __init__(self, args: Namespace, data_set: GeneralDataset,
                 writer: SummaryWriter, save_path: str, device: tc.device):
        # dataset, model, device, regularizer
        self.device = device
        self.data_set = data_set
        self.model = models.Model(args, data_set)
        self.regularizer = regularization.Regularizer(args)
        self.to_device()

        # estimate forcing interval
        if args.estimate_forcing:
            print(f"Estimating forcing interval...")
            tau_ac = estimate_forcing_interval(data_set.data.cpu().numpy(),
                                               False, mode="ACORR")[0]
            tau_mi = estimate_forcing_interval(data_set.data.cpu().numpy(),
                                               False, mode="MI")[0]
            mn = min(tau_ac, tau_mi)
            print(f"Estimated forcing interval: min(AC {tau_ac}, MI {tau_mi}) ---> {mn}")
            self.tau = mn
        else:
            print(f"Forcing interval set by user: {args.teacher_forcing_interval}")
            self.tau = args.teacher_forcing_interval

        # optimizer
        self.optimizer = optim.RAdam(self.model.parameters(), args.learning_rate)
        
        # others
        self.n_epochs = args.n_epochs
        self.gradient_clipping = args.gradient_clipping
        self.writer = writer
        self.use_reg = args.use_reg
        self.saver = saving.Saver(writer, save_path, args, self.data_set, self.regularizer)
        self.save_step = args.save_step
        self.loss_fn = nn.MSELoss()
        self.noise_level = args.noise_level

        # scheduler
        e = args.n_epochs
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [int(0.1*e), int(0.8*e), int(0.9*e)], 0.1)

    def to_device(self) -> None:
        '''
        Moves members to computing device.
        '''
        self.model.to(self.device)
        self.data_set.to(self.device)
        self.regularizer.to(self.device)

    def compute_loss(self, pred: tc.Tensor, target: tc.Tensor) -> tc.Tensor:
        '''
        Compute Loss w/ optional MAR loss.
        '''
        loss = .0
        loss += self.loss_fn(pred, target)

        if self.use_reg:
            lat_model_parameters = self.model.latent_model.get_latent_parameters()
            loss += self.regularizer.loss(lat_model_parameters)

        return loss

    def train(self):
        cum_T = 0.

        if self.gradient_clipping == 1:
            self.estimate_gc_norm()
        else:
            self.gc = self.gradient_clipping

        for epoch in range(1, self.n_epochs + 1):
            # enter training mode
            self.model.train()

            # measure time
            T_start = timer()

            # sample random sequences every epoch
            dataloader = self.data_set.get_rand_dataloader()
            for idx, (inp, target) in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                inp += tc.randn_like(inp) * self.noise_level
                pred = self.model(inp, self.tau)
                loss = self.compute_loss(pred, target)

                loss.backward()
                if self.gradient_clipping >= 1:
                    nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                             max_norm=self.gc)
                self.optimizer.step()

            self.scheduler.step()

            # timing
            T_end = timer()
            T_diff = T_end-T_start
            cum_T += T_diff
            cum_T_str = str(datetime.timedelta(seconds=cum_T)).split('.')[0]

            print(f"Epoch {epoch} took {round(T_diff, 2)}s | Cumulative time (h:mm:ss):" 
             f" {cum_T_str} | Loss = {loss.item()}")

        

            if epoch % self.save_step == 0:
                self.saver.epoch_save(self.model, epoch)


    def estimate_gc_norm(self):
        '''
        Estimate gradient clipping value as suggested by
        Pascanu, 2012: On the difficulty of training Recurrent Neural Networks.
        https://arxiv.org/abs/1211.5063
        Tracks gradient norms across 10 epochs of training and 
        computes the mean. GC clipping value is then set to 5 times
        that value.
        '''
        print("Estimating Gradient Clipping Value ...")
        params = deepcopy(self.model.state_dict())
        N_samples = 0
        running_g_norm = 0.
        for e in range(15):
            dataloader = self.data_set.get_rand_dataloader()
            for _, (inp, target) in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(inp, self.tau)
                loss = self.compute_loss(pred, target)

                loss.backward()
                if e > 5:
                    running_g_norm += nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                            max_norm=1e10)
                    N_samples += 1
                self.optimizer.step()
        
        gc_estimate = 5 * running_g_norm / N_samples
        self.gc = gc_estimate
        self.model.load_state_dict(params)
        print(f"Estimated Gradient Clipping Value: {self.gc}")