import torch as tc
from torch import nn
from sgvb import helpers as h
import torch.nn.functional as F
import math


# tc.set_default_tensor_type(tc.DoubleTensor)

#simple parallelized estimation of mean and covariance of the approximate posterior distribution
class DiagonalCovariance(nn.Module):
    def __init__(self, dim_x, dim_z):
        super(DiagonalCovariance, self).__init__()

        self.d_x = dim_x
        self.d_z = dim_z
        self.w_filter = nn.Parameter(tc.zeros(1), requires_grad=True)
        self.mean = nn.Linear(self.d_x, self.d_z, bias=False)
        self.logvar = nn.Linear(self.d_x, self.d_z, bias=True)

    def filter(self, x):
        xm1 = tc.cat((x[0:1, :], x), dim=0)[:-1]
        xp1 = tc.cat((x, x[-1, :].unsqueeze(0)), dim=0)[1:]
        return self.w_filter * xm1 + (1 - 2 * self.w_filter) * x + self.w_filter * xp1

    def get_sample(self, mean, log_sqrt_var):
        sample = mean + tc.exp(log_sqrt_var) * tc.randn(mean.shape[0], self.d_z)
        return sample

    def get_entropy(self, log_sqrt_var):
        entropy = tc.sum(log_sqrt_var) / log_sqrt_var.shape[0]
        return entropy

    def forward(self, x):
        x = x.view(-1, self.d_x)
        x = self.filter(x)

        mean = self.mean(x)
        log_sqrt_var = self.logvar(x)

        sample = self.get_sample(mean, log_sqrt_var)
        entropy = self.get_entropy(log_sqrt_var)

        return sample, entropy

#Estimation of mean and covariance of the approximate posterior distribution based on CNNs
class StackedConvolutions(nn.Module):
    def __init__(self, dim_x, dim_z, kernel_size=[3, 3, 3, 3], stride=[1], padding=[1], num_convs=(3, 1)):
        super(StackedConvolutions, self).__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z

        assert (len(kernel_size) == num_convs[0] + num_convs[1])
        assert (len(kernel_size) == len(stride) or len(stride) == 1)
        assert (len(kernel_size) == len(padding) or len(padding) == 1)

        if len(stride) == 1:
            stride *= len(kernel_size)
        if len(padding) == 1:
            padding *= len(kernel_size)

        mean_convs = []
        for i in range(num_convs[0]):
            mean_convs.append(nn.Conv1d(
                in_channels=dim_x if i == 0 else dim_z,
                out_channels=dim_z,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i]
            ))
        self.mean_conv = nn.Sequential(*mean_convs)

        logvar_convs = []
        for i in range(num_convs[1]):
            logvar_convs.append(nn.Conv1d(
                in_channels=dim_x if i == 0 else dim_z,
                out_channels=dim_z,
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i]
            ))
        self.logvar_conv = nn.Sequential(*logvar_convs)

    def to_batch(self, x):
        x = x.T
        x = x.unsqueeze(0)
        return x

    def from_batch(self, x):
        x = x.squeeze(0)
        x = x.T
        return x

    def get_sample(self, mean, log_sqrt_var):
        sample = mean + tc.exp(log_sqrt_var) * tc.randn(mean.shape[0], self.dim_z)
        return sample

    def get_entropy(self, log_sqrt_var):
        entropy = tc.sum(log_sqrt_var) / log_sqrt_var.shape[0]
        return entropy

    def forward(self, x):
        x = self.to_batch(x)

        mean = self.mean_conv(x)
        log_sqrt_var = self.logvar_conv(x)

        mean = self.from_batch(mean)
        log_sqrt_var = self.from_batch(log_sqrt_var)

        sample = self.get_sample(mean, log_sqrt_var)
        entropy = self.get_entropy(log_sqrt_var)

        return sample, entropy

    @property
    def mean(self):
        return self.mean_conv

    @property
    def logvar(self):
        return self.logvar_conv
