# tests/test_bayes.py
import jax.numpy as jnp
import pytest

from bp_error.bayes import posterior_loop, posterior_vectorized, BayesianUpdate

from pydantic import BaseModel


def test_no_observations_returns_prior():
    prior = jnp.array([0.3, 0.7])
    likelihoods = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    observations = jnp.array([1,1,1,1,0])
    result = posterior_loop(prior, likelihoods, observations)
    assert result


def test_bayesian_update():
    BayesianUpdate(prior = jnp.array([0.4, 0.6]), likelihoods =  jnp.array([[0.7, 0.3], [0.3, 0.7]]))


