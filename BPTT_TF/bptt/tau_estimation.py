import numpy as np
from scipy.signal import argrelmin

def mi(x, y, bins=64):
    """
    <Function copied from https://github.com/manu-mannattil/nolitsa>
    Calculate the mutual information between two random variables.
    Calculates mutual information, I = S(x) + S(y) - S(x,y), between two
    random variables x and y, where S(x) is the Shannon entropy.
    Parameters
    ----------
    x : array
        First random variable.
    y : array
        Second random variable.
    bins : int
        Number of bins to use while creating the histogram.
    Returns
    -------
    i : float
        Mutual information.
    """
    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    # Convert frequencies into probabilities.  Also, in the limit
    # p -> 0, p*log(p) is 0.  We need to take out those.
    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    # Calculate the corresponding Shannon entropies.
    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y


def dmi(x, maxtau=1000, bins=64):
    """
    <Function copied from https://github.com/manu-mannattil/nolitsa>
    Return the time-delayed mutual information of x_i.
    Returns the mutual information between x_i and x_{i + t} (i.e., the
    time-delayed mutual information), up to a t equal to maxtau.  Based
    on the paper by Fraser & Swinney (1986), but uses a much simpler,
    albeit, time-consuming algorithm.
    Parameters
    ----------
    x : array
        1-D real time series of length N.
    maxtau : int, optional (default = min(N, 1000))
        Return the mutual information only up to this time delay.
    bins : int
        Number of bins to use while calculating the histogram.
    Returns
    -------
    ii : array
        Array with the time-delayed mutual information up to maxtau.
    Notes
    -----
    For the purpose of finding the time delay of minimum delayed mutual
    information, the exact number of bins is not very important.
    """
    N = len(x)
    maxtau = min(N, maxtau)

    ii = np.empty(maxtau)
    ii[0] = mi(x, x, bins)

    for tau in range(1, maxtau):
        ii[tau] = mi(x[:-tau], x[tau:], bins)

    return ii


def estimate_forcing_interval(X: np.ndarray, dimensionwise: bool = True, reduction: str = "median",
                              max_lag: int = 100, bins: int = 64, mode: str = "MI") -> np.ndarray:
    '''
    Estimate the forcing interval by either Mutual Information (MI) or the Autocorrelation (ACORR) of each
    dimension of the data matrix `X`.

    The `mode` parameter indicates which algorithm/metric is used to estimate the forcing interval.

    `mode = ACORR`:
    ---------------
        The optimal forcing interval is chosen as the time lag where the 
        autocorrelation either reaches its first local minimum or reaches 1/e.
    `mode = MI`:
    ------------
        The optimal forcing interval is arguably the time lag where the Mutual Information
        drops to its first local minimum.
    
    If `dimensionwise=True`, this function will
    return an estimate for the forcing interval for each dimension of `X`.
    Else, the forcing interval is determined by `reduction` of the forcing interval
    array (defaults to the median forcing interval across dimensions). 

    Args:
        X (np.ndarray): T x N data matrix
        dimensionwise (bool): if forcing interval per dimension is to be returned
        reduction (str): reduction type if dimensionswise is false
        max_lag (int): optimal forcing interval is searched to a maximum of this value
        bins (int): only has an effect on MI computation
        mode (str): either 'ACORR' or 'MI'
    
    Returns:
        (np.ndarray) Array of forcing intervals of length N for `dimensionwise=True`,
        else of length 1.
    '''
    assert mode.upper() in ["MI", "ACORR"]
    assert reduction.lower() in ["min", "median", "mean", "max"]

    # data shape
    T, N = X.shape

    # standardize
    X_ = X#(X - X.mean(0)) /  X.std(0)

    if mode.upper() == "ACORR":
        # compute autocorrelation function
        corrs = np.empty((max_lag, N))
        for tau in range(max_lag):
            T_sub = T - tau
            # compute autocorrelation with given lag for all dims (1 x N)
            corr = 1 / T * np.sum((X_[:T_sub] - X.mean(0)) * (X_[tau:]- X.mean(0)), 0) / X.var(0)
            corrs[tau] = corr
    
        # compute 1/e and local minimum criterion for all corrs
        taus = np.empty(N, dtype=np.int8)
        x_ax = np.arange(max_lag)
        for dim in range(N):
            autocorr = corrs[:, dim]
            #plt.plot(autocorr)
            # decay to 1/e check
            tau_exp_decay = max_lag - int(np.interp(np.exp(-1), autocorr[::-1], x_ax)) - 1
            # first local minimum check
            candidates = argrelmin(autocorr)[0]
            if candidates.size == 0:
                tau_local_min = np.inf
            else:
                tau_local_min = candidates[0]
            # be conservative in choosing tau
            tau_est = min(tau_exp_decay, tau_local_min)
            taus[dim] = tau_est
    else:
        taus = np.empty(N, dtype=np.int8)
        for dim in range(N):
            # compute mutual information
            mis = dmi(X_[:, dim], max_lag, bins=bins)
            #plt.plot(mis)
            # first local minimum is optimal tau
            mn = argrelmin(mis)[0][0]
            taus[dim] = mn

    if not dimensionwise:
        reduction_fn = getattr(np, reduction.lower())
        taus = reduction_fn(taus, keepdims=True).astype(int)
    
    return taus