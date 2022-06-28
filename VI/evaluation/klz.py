"""
Calculate the KLz measure for a model with Monte Carlo sampling, or with a variational approximation.
The attractor geometry is approximated by a GMM with Gaussian distribution at each data point.
"""
import torch as tc


def calc_kl_with_covariance_approximation(model, x):
    time_steps = min(len(x), 5000)
    x = x[:time_steps]

    mu_inf = get_posterior_mean(model.rec_model, x)
    cov_inf = get_posterior_covariance(model.rec_model, x)
    mu_gen = get_prior_mean(model.gen_model, time_steps)
    cov_gen = get_prior_covariance(model.gen_model, time_steps)

    kl_mc, _ = calc_kl_mc(mu_inf, cov_inf, mu_gen, cov_gen)
    return kl_mc


def calc_kl_with_unit_covariance(model, x):
    time_steps = min(len(x), 5000)
    x = x[:time_steps]

    scaling = 1
    mu_inf = get_posterior_mean(model.rec_model, x)
    cov_inf = scaling * tc.ones_like(mu_inf)
    mu_gen = get_prior_mean(model.gen_model, time_steps)
    cov_gen = scaling * tc.ones_like(mu_gen)

    kl_mc, _ = calc_kl_mc(mu_inf, cov_inf, mu_gen, cov_gen)
    return kl_mc


def get_posterior_mean(rec_model, x):
    return rec_model.mean(x)


def get_posterior_covariance(rec_model, x):
    log_sqrt_var = rec_model.logvar(x)
    return tc.exp(log_sqrt_var) ** 2


def get_prior_mean(gen_model, time_steps):
    t_sample = 100
    assert int(time_steps / t_sample) == time_steps / t_sample
    return gen_model.get_latent_time_series_repeat(time_steps=t_sample, n_repeat=int(time_steps/t_sample))


def get_prior_covariance(gen_model, time_steps):
    sigma_gen = tc.diag(gen_model.R_z ** 2)
    A, W, h = gen_model.get_latent_parameters()
    c = tc.inverse(tc.eye(A.size()[0]) - (A + W).T.matmul(A + W)).matmul(sigma_gen)
    return c.diag().repeat(time_steps, 1)


def calc_kl_var(mu_inf, cov_inf, mu_gen, cov_gen):
    """
    Variational approximation of KL divergence (eq. 20, Hershey & Olsen, 2007)
    """
    kl_posterior_posterior = kl_between_two_gaussians(mu_inf, cov_inf, mu_inf, cov_inf)
    kl_posterior_prior = kl_between_two_gaussians(mu_inf, cov_inf, mu_gen, cov_gen)

    denominator = tc.sum(tc.exp(-kl_posterior_posterior), dim=1)
    nominator = tc.sum(tc.exp(-kl_posterior_prior), dim=1)
    nominator, denominator, outlier_ratio = clean_from_outliers(nominator, denominator)
    kl_var = (tc.mean(tc.log(denominator), dim=0) - tc.mean(tc.log(nominator), dim=0))
    return kl_var, outlier_ratio


def kl_between_two_gaussians(mu0, cov0, mu1, cov1):
    """
    For every time step t in mu0 cov0, calculate the kl to all other time steps in mu1, cov1.
    """
    T, n = mu0.shape

    cov1inv_cov0 = tc.einsum('tn,dn->tdn', cov0, 1 / cov1)  # shape T, T, n
    trace_cov1inv_cov0 = tc.sum(cov1inv_cov0, dim=-1)  # shape T,

    diff_mu1_mu0 = mu1.reshape(1, T, n) - mu0.reshape(T, 1, n)  # subtract every possible combination
    mahalonobis = tc.sum(diff_mu1_mu0 / cov1 * diff_mu1_mu0, dim=2)

    det1 = tc.prod(cov1, dim=1)
    det0 = tc.prod(cov0, dim=1)
    logdiff_det1det0 = tc.log(det1).reshape(1, T) - tc.log(det0).reshape(T, 1)

    kl = 0.5 * (logdiff_det1det0 - n + trace_cov1inv_cov0 + mahalonobis)
    return kl


def calc_kl_mc(mu_inf, cov_inf, mu_gen, cov_gen):
    mc_n = 1000
    t = tc.randint(0, mu_inf.shape[0], (mc_n,))

    std_inf = tc.sqrt(cov_inf)
    std_gen = tc.sqrt(cov_gen)

    z_sample = (mu_inf[t] + std_inf[t] * tc.randn(mu_inf[t].shape)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, std_inf)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = tc.mean(tc.log(posterior) - tc.log(prior), dim=0)
    return kl_mc, outlier_ratio


def clean_from_outliers(prior, posterior):
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    outlier_ratio = (1 - nonzeros.float()).mean()
    return prior, posterior, outlier_ratio


def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    precision = 1 / (std ** 2)
    precision = tc.diag_embed(precision)

    prec_vec = tc.einsum('zij,azj->azi', precision, vec)
    exponent = tc.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = tc.prod(std, dim=1)
    likelihood = tc.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T

