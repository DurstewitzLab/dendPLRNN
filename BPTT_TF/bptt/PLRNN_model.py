from typing import Optional, Tuple
from bptt.dataset import GeneralDataset
import torch.nn as nn
import torch as tc
import math
from torch.linalg import pinv


class PLRNN(nn.Module):
    """
    Piece-wise Linear Recurrent Neural Network (Durstewitz 2017)

    Args:
        dim_x: Dimension of the observations
        dim_z: Dimension of the latent states (number of hidden neurons)
        n_bases: Number of bases to use in the BE-PLRNN
        clip_range: latent state clipping value
        latent_model: Name of the latent model to use. Has to be in LATENT_MODELS
        mean_centering: Use mean centering
    """

    LATENT_MODELS = ['PLRNN', 'clipped-PLRNN', 'dendr-PLRNN']

    def __init__(self, dim_x: int, dim_z: int, n_bases: int, clip_range: float,
                 latent_model: str, mean_centering: bool, dataset: GeneralDataset):
        super(PLRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.n_bases = n_bases
        self.use_bases = False

        if latent_model == 'PLRNN':
            if n_bases > 0:
                print("Chosen model is vanilla PLRNN, the bases Parameter has no effect here!")
            self.latent_step = PLRNN_Step(dz=self.d_z, clip_range=clip_range, layer_norm=mean_centering)
        else:
            if latent_model == 'clipped-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for clipped-PLRNN!"
                self.latent_step = PLRNN_Clipping_Step(self.n_bases, dz=self.d_z, clip_range=clip_range,
                                                       layer_norm=mean_centering, dataset=dataset)
                self.use_bases = True
            elif latent_model == 'dendr-PLRNN':
                assert n_bases > 0, "n_bases has to be a positive integer for dendr-PLRNN!"
                self.latent_step = PLRNN_Basis_Step(self.n_bases, dz=self.d_z, clip_range=clip_range, 
                                                    layer_norm=mean_centering, dataset=dataset)
                self.use_bases = True
            else:
                raise NotImplementedError(f"{latent_model} is not yet implemented. Use one of: {self.LATENT_MODELS}.")

    def get_latent_parameters(self):
        '''
        Split the AW matrix and return A, W, h.
        A is returned as a 1d vector!
        '''
        AW = self.latent_step.AW
        A = tc.diag(AW)
        W = AW - tc.diag(A)
        h = self.latent_step.h
        return A, W, h

    def get_basis_expansion_parameters(self):
        alphas = self.latent_step.alphas
        thetas = self.latent_step.thetas
        return alphas, thetas

    def get_parameters(self):
        params = self.get_latent_parameters()
        if self.use_bases:
            params += self.get_basis_expansion_parameters()
        return params

    def forward(self, x, tau=None, z0=None, B=None):
        '''
        Forward pass with observations interleaved every n-th step.
        Credit @Florian Hess
        '''

        # switch dimensions for performance reasons
        x_ = x.permute(1, 0, 2)
        T, b, dx = x_.size()

        # no interleaving obs. if n is not specified
        if tau is None:
            tau = T + 1

        # pre-compute pseudo inverse
        B_PI = None
        if B is not None:
            B_PI = pinv(B)

        # initial state
        if z0 is None:
            z = tc.randn(size=(b, self.d_z), device=x.device)
            z = self.teacher_force(z, x_[0], B_PI)
        else:
            z = z0

        # stores whole latent state trajectory
        Z = tc.empty(size=(T, b, self.d_z), device=x.device)
        # gather parameters
        params = self.get_parameters()
        for t in range(T):
            # interleave observation every n time steps
            if (t % tau == 0) and (t > 0):
                z = self.teacher_force(z, x_[t], B_PI)
            z = self.latent_step(z, *params)
            Z[t] = z

        return Z.permute(1, 0, 2)

    @tc.no_grad()
    def generate(self, T, data, z0=None, B=None):
        '''
        Generate a trajectory of T time steps given
        an initial condition z0. If no initial condition
        is specified, z0 is teacher forced.
        '''
        # holds the whole generated trajectory
        Z = tc.empty((T, 1, self.d_z), device=data.device)

        # pre-compute pseudo inverse
        B_PI = None
        if B is not None:
            B_PI = pinv(B)

        # initial condition
        if z0 is None:
            z = tc.randn((1, self.d_z), device=data.device)
            z = self.teacher_force(z, data[0], B_PI)
        else:
            z = z0

        Z[0] = z
        params = self.get_parameters()
        for t in range(1, T):
            Z[t] = self.latent_step(Z[t-1], *params)

        return Z.squeeze_(1)

    def teacher_force(self, z: tc.Tensor, x: tc.Tensor,
                      B_PI: Optional[tc.Tensor] = None) -> tc.Tensor:
        '''
        Apply teacher forcing to the latent state vector z.
        If B_PI is None, identity mapping is assumed the first
        dx entries of z are teacher forced. If B_PI is not None,
        z is estimated using the least-squares solution.
        '''
        if B_PI is not None:
            z = x @ B_PI.t()
        else:
            z[:, :self.d_x] = x
        return z


class Latent_Step(nn.Module):
    def __init__(self, dz, clip_range=None, layer_norm=False):
        super(Latent_Step, self).__init__()
        self.clip_range = clip_range
        #self.nonlinearity = nn.ReLU()
        self.dz = dz

        if layer_norm:
            self.norm = lambda z: z - z.mean(dim=1, keepdim=True)
        else:
            self.norm = nn.Identity()

    def init_AW_random_max_ev(self):
        AW = tc.eye(self.dz) + 0.1 * tc.randn(self.dz, self.dz)
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(AW)))
        return nn.Parameter(AW / max_ev, requires_grad=True)

    def init_uniform(self, shape: Tuple[int]) -> nn.Parameter:
        # empty tensor
        tensor = tc.empty(*shape)
        # value range
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)

    def init_thetas_uniform(self, dataset: GeneralDataset) -> nn.Parameter:
        '''
        Initialize theta matrix of the basis expansion models such that 
        basis thresholds are uniformly covering the range of the given dataset
        '''
        mn, mx = dataset.data.min().item(), dataset.data.max().item()
        tensor = tc.empty((self.dz, self.db))
        # -mx to +mn due to +theta formulation in the basis step formulation
        nn.init.uniform_(tensor, -mx, -mn)
        return nn.Parameter(tensor, requires_grad=True)

    def init_AW(self):
        '''
        Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network
        with ReLU Nonlinearity https://arxiv.org/abs/1511.03771.
        '''
        matrix_random = tc.randn(self.dz, self.dz)
        matrix_positive_normal = (1 / self.dz) * matrix_random.T @ matrix_random
        matrix = tc.eye(self.dz) + matrix_positive_normal
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(matrix)))
        matrix_spectral_norm_one = matrix / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def clip_z_to_range(self, z):
        if self.clip_range is not None:
            tc.clip_(z, -self.clip_range, self.clip_range)
        return z


class PLRNN_Step(Latent_Step):
    def __init__(self, *args, **kwargs):
        super(PLRNN_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))

    def forward(self, z, A, W, h):
        z_activated = tc.relu(self.norm(z))
        z = A * z + z_activated @ W.t() + h
        return self.clip_z_to_range(z)

class PLRNN_Basis_Step(Latent_Step):
    def __init__(self, db, dataset=None, *args, **kwargs):
        super(PLRNN_Basis_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))
        self.db = db

        if dataset is not None:
            self.thetas = self.init_thetas_uniform(dataset)
        else:
            self.thetas = nn.Parameter(tc.randn(self.dz, self.db))
        self.alphas = self.init_uniform((self.db, ))

    def forward(self, z, A, W, h, alphas, thetas):
        z_norm = self.norm(z).unsqueeze(-1)
        # thresholds are broadcasted into the added dimension of z
        be = tc.sum(alphas * tc.relu(z_norm + thetas), dim=-1)
        z = A * z + be @ W.t() + h
        return self.clip_z_to_range(z)

class PLRNN_Clipping_Step(Latent_Step):
    def __init__(self, db, dataset=None, *args, **kwargs):
        super(PLRNN_Clipping_Step, self).__init__(*args, **kwargs)
        self.AW = self.init_AW()
        self.h = self.init_uniform((self.dz, ))
        self.db = db
        if dataset is not None:
            self.thetas = self.init_thetas_uniform(dataset)
        else:
            self.thetas = nn.Parameter(tc.randn(self.dz, self.db))
        self.alphas = self.init_uniform((self.db, ))

    def forward(self, z, A, W, h, alphas, thetas):
        z_norm = self.norm(z).unsqueeze(-1)
        be_clip = tc.sum(alphas * (tc.relu(z_norm + thetas) - tc.relu(z_norm)), dim=-1)
        z = A * z + be_clip @ W.t() + h
        return z
