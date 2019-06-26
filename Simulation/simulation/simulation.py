# !pip install scipy==1.2  --upgrade
import numpy as np
import numpy.random as ra
from scipy.interpolate import interp1d
import statsmodels.distributions.empirical_distribution as edf
#from scipy.misc import *
import random


def linear_interpolation(array, sample_size):
    sample_edf = edf.ECDF(array)
    slope_changes = sorted(set(array))

    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]

    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)

    return inverted_edf(np.random.uniform(0, 1, sample_size))


def inverse_transform_sampling(data, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def numpy_random_choice(array, size, replace=True, prob=None):
    return np.random.choice(array, size=size, replace=replace, p=prob)


def random_choices(population, weights=None, cum_weights=None, k=1):
    return random.choices(population, weights=weights, cum_weights=cum_weights, k=k)


def quantile_weighted_random(array, quantile_steps=0.2, sample_size=10):

    quantile = np.quantile(array, q=np.arange(0, 1+quantile_steps, quantile_steps))

    rand_int = np.random.randint(0, len(quantile)-1, size=sample_size)

    return [np.random.uniform(quantile[x], quantile[x+1], 1)[0] for x in rand_int]






