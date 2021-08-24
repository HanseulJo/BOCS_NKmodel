"""
Function version of NKmodel
"""

import random
import numpy as np
import itertools

def binarr2int(arr):
    # Example:
    # np.array([0 1 1 0 1 0]) -> 26
    return int(arr.dot(1 << np.arange(arr.size)[::-1]))

def int2binarr(num, width):
    return np.array([int(x) for x in np.binary_repr(num, width=width)])

def dependency(N, K):  # interdependency matrix
    interdependence = (np.eye(N, dtype=float) == 1)
    for i in range(N):
        dependence = np.random.choice(list(set(range(N)) - set([i])), size=K, replace=False)
        interdependence[i][dependence] = True
    return interdependence

def contri(N, K):  # contribution map
    # each i-th row is a single contribution map for i-th position.
    return np.random.random(size=(N,2**(K+1)))  # floats [0, 1]

def _calculate_ith_contribution(state, ctrb_map, inter_mat, i):
    interdep = inter_mat[i].copy()
    interdep[i] = False
    label = np.append(state[i], state[interdep])  # the value of i-th locus should be the first entry of the 'label'.
    return ctrb_map[i][binarr2int(label)]

def fitness(state, ctrb_map, inter_mat, negative=False):
    """
    Given a state(: an array of length N), 
    Return fitness value followed by associated contributions.
    """
    state = state.flatten()
    N = state.size
    ctrbs = [_calculate_ith_contribution(state, ctrb_map, inter_mat, i) for i in range(N)]
    fitness_value = sum(ctrbs) / N  # averaged fitness value --> btw 0 ~ 1
    if negative:
        fitness_value = -fitness_value
    return ctrbs + [fitness_value]  # payoff

def NK_landscape(N=6, K=1, negative=False):
    inter_mat = dependency(N, K)
    ctrb_map = contri(N, K)
    states = itertools.product(range(2), repeat=N)
    landscape = [fitness(np.array(state), ctrb_map, inter_mat, negative=negative) for state in states]
    landscape = np.array(landscape)
    return inter_mat, ctrb_map, landscape

