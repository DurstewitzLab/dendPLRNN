class Annealer:
    def __init__(self, args):
        self.n_epochs = args.n_epochs
        self.annealing = args.annealing
        self.annealing_step_size = args.annealing_step_size

    def get_alpha(self, epoch):
        epoch = self.annealing_step_size * int(epoch / self.annealing_step_size)
        training_progress = epoch / self.n_epochs
        if self.annealing == 'None' or self.annealing is None:
            alpha = 0.5
        elif self.annealing == 'quad':
            alpha = training_progress * (1 - 0.5 * training_progress)
        elif self.annealing == 'lin':
            alpha = 0.5 * training_progress
        else:
            print('choose one of the following for annealing: None, quad, lin')
            raise NotImplementedError
        return alpha
