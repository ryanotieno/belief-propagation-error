import jax.numpy as jnp
from jax import Array
from jax import vmap
import math

def perc_message_passing(network, adj_mat, i, j, p, max_itr):
    failure_fwr = network[f'{j}, {i}']
    failure_bkw = network[f'{i}, {j}']
    for a in range(0, max_itr): #get me all of 0's neighbors
        if adj_mat[i][a] == 1 and a != j:
            #message pass into the neighbors of i, and bring back out their mu values
            failure_fwr =  (1-p) + (p*failure_fwr * network[f'{i}, {a}']) #message_passing(network, adj_mat, i, a, max_itr)
        
    for a in range(0, max_itr): #get me all of 2's neighbors
        if adj_mat[j][a] == 1 and a != i:
            #message pass into the neighbors of i, and bring back out their mu values
            failure_bkw =  (1-p) + (p*failure_bkw * network[f'{j}, {a}']) #message_passing(network, adj_mat, i, a, max_itr)
    failure = jnp.array([failure_fwr, failure_bkw])
    return failure

def declare_vector(p, N):
    V = []
    for i in range(N*(N-1)):
        V.append(math.log(1-p + p*0.5))
    return V

def update_v_and_messages(p, messages):
    V = []
    for keys, values in messages.items():
        i, j = keys.split(',')
        V.append(math.log(1-p + p*messages[f'{i},{j}']))
    return V

def perc_log_meesage_pass(A, B, V, messages):
    for keys, values in B.items():
        i, j = keys.split(',')
        temp_B = jnp.array(B[f'{i},{j}'])
        mu_i_j = math.exp(jnp.dot(temp_B, jnp.array(V)))
        if A[i][j] != 0:
            messages[f'{i},{j}'] = mu_i_j
        else:
            messages[f'{i},{j}'] = 0.5
    return messages
        











def initialize_messages(connections, A, N):
    for i in range(N):
        for j in range(N):
            if A[i][j] == 1 and A[j][i] == 1:
                connections[f'{i}, {j}'] = 0.5
                connections[f'{j}, {i}'] = 0.5
    return connections