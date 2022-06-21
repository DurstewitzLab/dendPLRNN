import torch as tc

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
    vec=vec.float()
    precision = 1 / (std ** 2)
    precision = tc.diag_embed(precision).float()

    prec_vec = tc.einsum('zij,azj->azi', precision, vec)
    exponent = tc.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = tc.prod(std, dim=1)
    likelihood = tc.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T




## KLX Statespace

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
    
    #print(mu_inf.shape)
    #print(std_inf.shape)

    z_sample = (mu_inf[t] + std_inf[t] * tc.randn(mu_inf[t].shape)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, std_inf)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = tc.mean(tc.log(posterior) - tc.log(prior), dim=0)
    return kl_mc, outlier_ratio



def calc_kl_from_data(mu_gen, data_true):
    
    time_steps = min(len(data_true), 10000)
    mu_inf= data_true[:time_steps]
    
    mu_gen=mu_gen[:time_steps]
    

    scaling = 1.
    cov_inf = tc.ones(data_true.shape[-1]).repeat(time_steps, 1)*scaling
    cov_gen = tc.ones(data_true.shape[-1]).repeat(time_steps, 1)*scaling

    kl_mc1, _  = calc_kl_mc(mu_gen, cov_gen.detach(), mu_inf.detach(), cov_inf.detach())

    kl_mc2, _  = calc_kl_mc(mu_inf.detach(), cov_inf.detach(), mu_gen, cov_gen.detach())

    kl_mc =1 / 2 * (kl_mc1 + kl_mc2)

    #scaling = 1
   # mu_inf = get_posterior_mean(model.rec_model, x)
    #cov_true = scaling * tc.ones_like(data_true)
   # mu_gen = get_prior_mean(model.gen_model, time_steps)
    #cov_gen = scaling * tc.ones_like(data_gen)

    #kl_mc, _ = calc_kl_mc(data_true, cov_true, data_gen, cov_gen)
    return kl_mc