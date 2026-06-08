import jax.numpy as jnp
from jax import Array


def posterior_loop(
    prior,
    likelihoods,
    observations,
):
    posterior = prior

    for obs in observations:
        likelihood = likelihoods[:, obs]

        temp = posterior * likelihood

        posterior = temp / jnp.sum(temp)

    return posterior


def posterior_vectorized(
    prior: Array,
    likelihoods: Array,
    observations: Array,
) -> Array:

    observation_likelihoods = likelihoods[:, observations]

    total_likelihood = jnp.prod(observation_likelihoods, axis=1)

    temp = prior * total_likelihood

    posterior = temp / jnp.sum(temp)

    return posterior