import jax
import jax.numpy as jnp
import matplotlib as plt

from bp_error.bayes import posterior_loop, posterior_vectorized

# Two-urn setup
prior = jnp.array([0.5, 0.5])  # [P(A), P(B)]
likelihoods = jnp.array([
    [0.7, 0.3],  # urn A: P(W|A)=0.7, P(K|A)=0.3
    [0.3, 0.7],  # urn B
])

# Generate 100 draws from urn A
key = jax.random.PRNGKey(0)
# TODO: use jax.random.choice with probabilities [0.7, 0.3] to generate observations.
observations = jnp.array(jax.random.choice(jnp.array(key), a = jnp.array([0, 1]), shape = (5,), p = jnp.array([0.7, 0.3])))
# TODO: for n in 1..100, compute the posterior after the first n observations,
# and store P(A | data) in a list.
results = posterior_loop(prior, likelihoods, observations)
# TODO: plot P(A | data) vs. n.