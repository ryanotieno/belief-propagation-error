# src/bp_error/bayes.py
import jax.numpy as jnp
from jax import Array
from jax import vmap


def posterior_loop(
    prior: Array,
    likelihoods: Array,
    observations: Array,
) -> Array:
    """Sequential Bayesian update via an explicit Python loop.

    Parameters
    ----------
    prior : Array, shape (K,)
        Prior probability over K hypotheses. Must sum to 1
    likelihoods : Array, shape (K, M)
        likelihoods[k, m] = P(observation = m | hypothesis = k).
    observations : Array, shape (T,)
        Sequence of T observation indices in {0, ..., M-1}.

    Returns
    -------
    posterior : Array, shape (K,)
        Posterior over hypotheses after all T observations.
    """
    if jnp.sum(prior) != 1:
        return 1
    if len(observations) == 0:
        return prior
    temp = prior
    results = []
    results.append(temp)
    for obs in observations:
        temp = temp * likelihoods[:, obs]
        temp = temp / jnp.sum(temp)
        results.append(temp)
    return results
    ...


def posterior_vectorized(
    prior: Array,
    likelihoods: Array,
    observations: Array,
) -> Array:
    """Sequential Bayesian update via array operations, no Python loop.

    Same semantics as posterior_loop. Hint: collect the likelihood of each
    observation as a (T, K) array, take the product along T, multiply by
    prior, normalize.
    """
    # TODO: implement
    likelihoods_stack = likelihoods[:, observations]

    likelihoods_stack = jnp.array(likelihoods_stack)
    #so what I want now is to just multiply urn a by everything in the first column, and row b by the second.
    #first i will compress my stack using prod()
    stack_comp = jnp.prod(likelihoods_stack, axis = 1)
    posterior = prior * stack_comp
    posterior = posterior / jnp.sum(posterior)
    return posterior