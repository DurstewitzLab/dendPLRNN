import torch as tc
import matplotlib.pyplot as plt

def marginalize_pdf(pdf, except_dims):
    """
    Marginalize out all except the specified dims
    :param pdf: multidimensional pdf
    :param except_dims: specify dimensions to keep
    :return: marginalized pdf
    """
    if len(pdf.shape) > 2:
        l = list(range(len(pdf.shape)))
        l = [i for i in l if i not in except_dims]
        pdf = pdf.sum(tuple(l))
    return pdf


def plot_kl(x_gen, x_true, n_bins):
    p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true, n_bins)
    kl_value = kullback_leibler_divergence(p_true, p_gen)
    p_true = marginalize_pdf(p_true, except_dims=(0, 2))
    if p_gen is not None:
        p_gen = marginalize_pdf(p_gen, except_dims=(0, 2))
    else:
        p_gen = 0 * p_true
    if kl_value is None:
        kl_string = 'None'
    else:
        kl_string = '{:.2f}'.format(kl_value)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(p_gen.numpy().T[::-1])
    axs[0].set_xticks(())
    axs[0].set_yticks(())
    axs[0].set_title('KLx: {}'.format(kl_string))
    axs[1].imshow(p_true.numpy().T[::-1])
    axs[1].set_xticks(())
    axs[1].set_yticks(())
    axs[1].set_title('data')
    plt.show()


def loss_kl(x1, x2, n_bins=10, symmetric=False):
    """
    Spatial KL-divergence loss function
    :param x1: time series 1
    :param x2: time series 2, reference time series
    :param n_bins: number of histogram bins
    :param symmetric: symmetrized KL-divergence
    :return: loss (scalar)
    """
    p1, p2 = get_pdf_from_timeseries(x1, x2, n_bins)
    kl21 = kullback_leibler_divergence(p2, p1)

    if not symmetric:
        loss = kl21  # assuming p2 is ground truth
    else:
        kl12 = kullback_leibler_divergence(p1, p2)
        loss = (kl12 + kl21) / 2
    return loss


def kullback_leibler_divergence(p1, p2):
    """
    Calculate Kullback-Leibler divergence
    """
    if p1 is None or p2 is None:
        kl = tc.tensor([float('nan')])
    else:
        kl = (p1 * tc.log(p1 / p2)).sum()
    return kl


def calc_histogram(x, n_bins, min_, max_):
    """
    Calculate a multidimensional histogram in the range of min and max
    works by aggregating values in sparse tensor,
    then exploits the fact that sparse matrix indices may contain the same coordinate multiple times,
    the matrix entry is then the sum of all values at the coordinate
    for reference: https://discuss.pytorch.org/t/histogram-function-in-pytorch/5350/9
    Outliers are discarded!
    :param x: multidimensional data: shape (N, D) with N number of entries, D number of dims
    :param n_bins: number of bins in each dimension
    :param min_: minimum value
    :param max_: maximum value to consider for histogram
    :return: histogram
    """
    dim_x = x.shape[1]  # number of dimensions

    coordinates = (n_bins * (x - min_) / (max_ - min_)).long()

    # discard outliers
    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    indices = tc.ones(coordinates.shape[0], device=coordinates.device)
    if 'cuda' == coordinates.device.type:
        tens = tc.cuda.sparse.FloatTensor
    else:
        tens = tc.sparse.FloatTensor
    return tens(coordinates.t(), indices, size=size_).to_dense()


def get_min_max_range(x_true):
    std = x_true.std(0)
    return -2 * std, 2 * std


def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf


def get_pdf_from_timeseries(x_gen, x_true, n_bins):
    """
    Calculate spatial pdf of time series x1 and x2
    :param x_gen: multivariate time series: shape (T, dim)
    :param x_true: multivariate time series, used for choosing range of histogram
    :param n_bins: number of histogram bins
    :return: pdfs
    """
    min_, max_ = get_min_max_range(x_true)
    hist_gen = calc_histogram(x_gen, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return p_gen, p_true


def klx_metric(x_gen, x_true, n_bins=30):
    # plot_kl(x_gen, x_true, n_bins)
    p_gen, p_true = get_pdf_from_timeseries(x_gen, x_true, n_bins)
    return kullback_leibler_divergence(p_true, p_gen)


