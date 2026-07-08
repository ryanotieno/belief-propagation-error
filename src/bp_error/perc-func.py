import jax.numpy as jnp
from jax import Array
from jax import vmap

def perc_message_passing(network, adj_mat, i, j, p, max_itr):
    failure_fwr = network[f'{j}, {i}']
    failure_bkw = network[f'{i}, {j}']
    for a in range(0, max_itr): #get me all of 0's neighbors
        if adj_mat[i][a] == 1 and a != j:
            #message pass into the neighbors of i, and bring back out their mu values
            failure_fwr =  failure_fwr * network[f'{i}, {a}'] #message_passing(network, adj_mat, i, a, max_itr)
        
    for a in range(0, max_itr): #get me all of 2's neighbors
        if adj_mat[j][a] == 1 and a != i:
            #message pass into the neighbors of i, and bring back out their mu values
            failure_bkw =  failure_bkw * network[f'{j}, {a}'] #message_passing(network, adj_mat, i, a, max_itr)
    failure = jnp.array([(1-p) + (p*failure_fwr), (1-p) + (p*failure_bkw)])
    return failure