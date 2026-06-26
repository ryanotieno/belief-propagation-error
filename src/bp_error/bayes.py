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

from pydantic import BaseModel, ConfigDict, field_validator


class BayesianUpdate(BaseModel):
    """A single Bayesian inference problem."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prior: Array
    likelihoods: Array  # shape (K, M)

    @field_validator("prior")
    @classmethod
    def prior_sums_to_one(cls, v: Array) -> Array:
        # TODO: raise ValueError if v does not sum to 1 (within tolerance).
        if jnp.sum(v) < 0.98 or jnp.sum(v) > 1:
            raise ValueError("prior has to sum to 1")
        return v
        ...

    @field_validator("likelihoods")
    @classmethod
    def likelihood_rows_sum_to_one(cls, v: Array) -> Array:
        # TODO: raise ValueError if any row does not sum to 1.
        for like in v:
            if jnp.sum(like) < 0.98 or jnp.sum(like) > 1:
                raise ValueError("likelihoods has to sum to 1")


        ...

    def update(self, observations: Array) -> Array:
        """Return the posterior after observing the sequence."""
        # TODO: call posterior_vectorized.
        result = posterior_vectorized(self, observations)
        print(result)
        return result
        ...

def grid_posterior_loop(grid, prior, observations):
    results = []
    results.append(prior)
    posterior = prior
    for obs in observations:
        if obs == 0:
            likelihood = grid
        else:
            likelihood = 1 - grid
        posterior = likelihood * posterior
        posterior = posterior / jnp.sum(posterior)
        results.append(posterior)

    results = jnp.array(results)
    results = jnp.prod(results, axis = 0)
    return posterior