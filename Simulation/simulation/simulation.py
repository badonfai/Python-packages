"""
Small list of functions to simulate data from 1 Dimensional Array

"""


# !pip install scipy==1.2  --upgrade
import numpy as np
from scipy.interpolate import interp1d
import statsmodels.distributions.empirical_distribution as edf
import random


def linear_interpolation(array, sample_size):
    """
    Sampling from 1D array with linear interpolation as inverse cdf
    :param array: input 1D array of Numbers
    :param sample_size: number of samples
    :return: array of simulated data
    """
    sample_edf = edf.ECDF(array)
    slope_changes = sorted(set(array))
    sample_edf_values_at_slope_changes = [sample_edf(item) for item in slope_changes]
    inverted_edf = interp1d(sample_edf_values_at_slope_changes, slope_changes)
    return inverted_edf(np.random.uniform(0, 1, sample_size))


def inverse_transform_sampling(array, n_bins=40, n_samples=1000):
    """
    Sampling from 1D array with an alternative implementation linear interpolation as inverse cdf
    :param array: input 1D array of Numbers
    :param n_bins:  it defines the number of equal-width bins in the given range
    :param n_samples: the number of samples
    :return: array of simulated data
    """
    hist, bin_edges = np.histogram(array, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def numpy_random_choice(array, size, replace=True, prob=None):
    """
    Using numpy package to choose random number of samples from an array
    :param array: input 1D array of Numbers
    :param size: the number of  samples
    :param replace: Whether the sample is with or without replacement
    :param prob: The probabilities associated with each entry
    :return: array of simulated data
    """
    return np.random.choice(array, size=size, replace=replace, p=prob)


def random_choices(population, weights=None, cum_weights=None, k=1):
    """
    Using standard library "random" to generate data from an Array
    :param population: input 1D array of Numbers
    :param weights:  a weights sequence is specified, selections are made according to the relative weights.
    :param cum_weights: can use any numeric type that interoperates
    :param k: Return a k sized list of elements chosen from the population with replacement.
    :return: array of simulated data
    """
    return random.choices(population, weights=weights, cum_weights=cum_weights, k=k)


def quantile_weighted_random(quantile, probs, sample_size=100):
    """
    Using qunatiles to generate from the CDF data samples
    :param array: input 1D array of Numbers
    :param quantile_steps: the step number for the quantiles
    :param sample_size: number of samples to generate
    :return: array of simulated data
    """
    if len(quantile) < 2:
        raise Exception("input has less than 2 elements")

    if np.sum(probs) != 1:
        raise Exception("the probability vector does not add up to 1")

    rand_int = random.choices(range(0, len(quantile)-1), weights=probs, k=sample_size)

    return [np.random.uniform(quantile[x], quantile[x+1], 1)[0] for x in rand_int]
