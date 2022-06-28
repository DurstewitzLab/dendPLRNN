import torch.nn as nn
import torch as tc
import math


def get_input(inputs, t):
    if inputs is not None:
        s = inputs[t:t + 1]
    else:
        s = None
    return s

def mahalonobis_distance(residual, matrix):
    return - 0.5 * (residual.t() @ residual * tc.inverse(tc.diag(matrix ** 2))).sum()


def log_det(diagonal_matrix):
    return - tc.log(diagonal_matrix.sum())


class PLRNN(nn.Module):
    """
    Piece-wise Linear Recurrent Neural Network (Durstewitz 2017)
    """

    def __init__(self, dim_x, dim_z, dim_s, n_bases, n_batches, clip_range):
        super(PLRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.d_s = dim_s
        self.n_bases = n_bases
        self.n_batches = n_batches

        # constrains covariance to be positive semi-definite
        # self.log_R_x = nn.Parameter(tc.zeros(self.d_x), requires_grad=True)
        # self.log_R_z = nn.Parameter(tc.zeros(self.d_z), requires_grad=True)
        # self.log_R_z0 = nn.Parameter(tc.zeros(self.d_z), requires_grad=True)
        # self.R_x = tc.exp(self.log_R_x)
        # self.R_z = tc.exp(self.log_R_z)
        # self.R_z0 = tc.exp(self.log_R_z0)

        self.R_x = nn.Parameter(tc.ones(self.d_x), requires_grad=True)
        self.R_z = nn.Parameter(tc.ones(self.d_z), requires_grad=True)
        self.R_z0 = nn.Parameter(tc.ones(self.d_z), requires_grad=True)

        self.z0 = nn.Parameter(tc.randn(self.n_batches, self.d_z), requires_grad=True)

        use_no_bases = (self.n_bases is None or self.n_bases == 0)
        if use_no_bases:
            self.latent_step = PLRNN_Step(self.d_z, self.d_s, clip_range=clip_range)
        else:
           # self.latent_step = PLRNN_Basis_Step(self.d_z, self.n_bases, self.d_s, clip_range=clip_range)
            self.latent_step = PLRNN_Clipping_Step(self.d_z, self.n_bases, self.d_s, clip_range=clip_range)
        self.observation = linear_observation(self.d_z, self.d_x)

    def get_latent_parameters(self):
        AW = self.latent_step.AW
        A = tc.diag(AW)
        W = AW - tc.diag(A)
        h = self.latent_step.h
        return A, W, h

    def get_initial_state(self, z0):
        if z0 is None:
            z0 = tc.randn(1, self.d_z)
        return z0

    def get_latent_time_series(self, time_steps=1000, z0=None, inputs=None):
        z = self.get_initial_state(z0)
        Z = []
        for t in range(time_steps):
            z = (self.latent_step(z, get_input(inputs, t)))
            Z.append(z)
        Z = tc.cat(Z)
        return Z

    def get_observed_time_series(self, time_steps=1000, z0=None, inputs=None):
        Z = self.get_latent_time_series(time_steps, z0, inputs)
        X = self.observation(Z)
        return X

    def get_latent_time_series_repeat(self, time_steps=1000, n_repeat=10):
        with tc.no_grad():
            cut_off = 1000
            Z = []
            z = tc.randn(n_repeat, self.d_z)
            for t in range(time_steps + cut_off):
                z = self.latent_step(z, None)
                Z.append(z)
            Z = Z[cut_off:]
            Z = tc.stack(Z, dim=1)
            shape = (n_repeat * time_steps, self.d_z)
            Z = tc.reshape(Z, shape)
        return Z

    def get_observed_time_series_repeat(self, time_steps=1000, n_repeat=10):
        Z = self.get_latent_time_series_repeat(time_steps, n_repeat)
        X = self.observation(Z)
        return X

    def log_likelihood(self, x, z, s, batch_index):

        def mahalonobis_distance(residual, matrix):
            return - 0.5 * (residual.t() @ residual * tc.inverse(tc.diag(matrix ** 2))).sum()

        distance_x = mahalonobis_distance(x - self.observation(z), self.R_x)
        distance_z0 = mahalonobis_distance((tc.squeeze(self.z0[batch_index, :]) - tc.squeeze(z[0, :])).unsqueeze(0),
                                           self.R_z0)
        distance_z = mahalonobis_distance(z[1:, :] - self.latent_step(z[:-1, :], s=s), self.R_z)

        def log_det(diagonal_matrix):
            return - tc.log(diagonal_matrix).sum()

        time_steps = x.shape[0]
        constant_z = - 0.5 * self.d_z * tc.log(tc.tensor(2 * math.pi)) * time_steps
        constant_x = - 0.5 * self.d_x * tc.log(tc.tensor(2 * math.pi)) * time_steps
        ll_z = distance_z0 + distance_z + log_det(self.R_z) * (time_steps - 1) + log_det(self.R_z0) + constant_z
        ll_x = distance_x + log_det(self.R_x) * time_steps + constant_x

        return ll_x / time_steps, ll_z / time_steps


class Observation(nn.Module):
    def __init__(self, dz, dx):
        super(Observation, self).__init__()
        self.dz = dz
        self.dx = dx
        self.B = nn.Parameter(tc.randn(self.dz, self.dx), requires_grad=True)


class linear_observation(Observation):
    def __init__(self, dz, dx):
        super(linear_observation, self).__init__(dz, dx)

    def forward(self, z):
        return tc.einsum('zx,bz->bx', (self.B, z))


class linear_relu_observation(Observation):
    def __init__(self, dz, dx):
        super(linear_relu_observation, self).__init__(dz, dx)

    def forward(self, z):
        z = tc.relu(z)
        return tc.einsum('zx,bz->bx', (self.B, z))

class Latent_Step(nn.Module):
    def __init__(self, clip_range, dz, ds):
        super(Latent_Step, self).__init__()
        self.clip_range = clip_range
        self.nonlinearity = nn.LeakyReLU(0.18)
        self.dz = dz
        self.ds = ds

        if self.ds is not None:
            self.C = nn.Parameter(tc.randn(self.dz, self.ds), requires_grad=True)

    def init_AW_random_max_ev(self):
        AW = tc.eye(self.dz) + 0.1 * tc.randn(self.dz, self.dz)
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(AW)[0]))
        return nn.Parameter(AW / max_ev, requires_grad=True)

    def init_AW(self):
        # from: Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network with ReLU Nonlinearity
        matrix_random = tc.randn(self.dz, self.dz)
        matrix_positive_normal = 1 / (self.dz * self.dz) * matrix_random @ matrix_random.T
        matrix = tc.eye(self.dz) + matrix_positive_normal
        max_ev = tc.max(tc.abs(tc.linalg.eigvals(matrix)[0]))
        matrix_spectral_norm_one = matrix / max_ev
        return nn.Parameter(matrix_spectral_norm_one, requires_grad=True)

    def clip_z_to_range(self, z):
        if self.clip_range is not None:
            z = tc.max(z, -self.clip_range * tc.ones_like(z))
            z = tc.min(z, self.clip_range * tc.ones_like(z))
        return z

    def add_input(self, s):
        if s is not None:
            input = tc.einsum('ij,bj->bi', (self.C, s))
            if input.shape[0] > 1:  # for batch-wise processing
                input = input[1:]
        else:
            input = 0
        return input


class PLRNN_Step(Latent_Step):
    def __init__(self, dz, ds=None, clip_range=None):
        super(PLRNN_Step, self).__init__(clip_range=clip_range, dz=dz, ds=ds)
        self.AW = self.init_AW()
        self.h = nn.Parameter(tc.randn(self.dz), requires_grad=True)

    def forward(self, z, s=None):
        A, W = split_diag_offdiag(self.AW)
        z = tc.einsum('iz,bz->bi', (A, z)) + tc.einsum('iz,bz->bi',
                                                       (W, self.nonlinearity(z))) + self.h + self.add_input(s)
        return self.clip_z_to_range(z)


class PLRNN_Basis_Step(Latent_Step):
    def __init__(self, dz, db, ds=None, clip_range=None):
        super(PLRNN_Basis_Step, self).__init__(clip_range=clip_range, dz=dz, ds=ds)
        self.AW = self.init_AW()
        self.h = nn.Parameter(tc.randn(self.dz), requires_grad=True)
        self.db = db
        self.thetas = nn.Parameter(2 * tc.randn(self.dz, self.db) - 1, requires_grad=True)
        self.alphas = nn.Parameter(2 * tc.randn(self.db) - 1, requires_grad=True)

    def forward(self, z, s=None):
        A, W = split_diag_offdiag(self.AW)
        z_temp = z.unsqueeze(-1).repeat(1, 1, self.db)  # shape: batch, dz, basis
        basis_expansion = tc.sum(tc.mul(self.alphas, self.nonlinearity(z_temp - self.thetas)), dim=-1)
        z = tc.einsum('iz,bz->bi', (A, z)) + tc.einsum('iz,bz->bi', (W, basis_expansion)) + self.h + self.add_input(s)
        return self.clip_z_to_range(z)


class PLRNN_Clipping_Step(Latent_Step):
    def __init__(self, dz, db, ds=None, clip_range=None):
        super(PLRNN_Clipping_Step, self).__init__(clip_range=clip_range, dz=dz, ds=ds)
        self.AW = self.init_AW()
        self.h = nn.Parameter(tc.randn(self.dz), requires_grad=True)
        self.db = db
        self.thetas = nn.Parameter(2 * tc.randn(self.dz, self.db) - 1, requires_grad=True)
        self.alphas = nn.Parameter(2 * tc.randn(self.db) - 1, requires_grad=True)
        self.clipping = nn.Parameter(5 * tc.ones(self.dz, self.db), requires_grad=False)

    def forward(self, z, s=None):
        A, W = split_diag_offdiag(self.AW)
        z_temp = z.unsqueeze(-1).repeat(1, 1, self.db)  # shape: batch, dz, basis
        basis_expansion = tc.sum(tc.mul(self.alphas, self.nonlinearity(z_temp - self.thetas)), dim=-1) - tc.sum(
            tc.mul(self.alphas, self.nonlinearity(z_temp - self.thetas + self.clipping)), dim=-1)
        z = tc.einsum('iz,bz->bi', (A, z)) + tc.einsum('iz,bz->bi', (W, basis_expansion)) + self.h + self.add_input(s)
        return z


def orthogonalize(A):
    """Find close orthogonal matrix, by dropping diagonal S in SVD decomposition (i.e. set EV to 1)"""
    U, S, V = tc.svd(A)
    return U @ V.T


def split_diag_offdiag(A):
    diag = tc.diag(tc.diag(A))
    off_diag = A - diag
    return diag, off_diag