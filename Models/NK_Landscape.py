import random
import numpy as np
import itertools

def dependency(N, K):  # interdependency matrix
    interdependence = (np.eye(N, dtype=float) == 1)
    for i in range(N):
        dependence = np.random.choice(list(set(range(N)) - set([i])), size=K, replace=False)
        interdependence[i][dependence] = True
    return interdependence

def contri(N, K):  # contribution map
    contributions = [{} for _ in range(N)]
    for i in range(N):
        for label in itertools.product(range(2), repeat=K+1):  # K+1 subcollection of loci values that effects the locus i
            contributions[i][label] = random.random()  # float [0, 1]
    return contributions

def _calculate_ith_contribution(state, ctrb_map, inter_mat, i):
    interdep = inter_mat[i].copy()
    interdep[i] = False
    label = tuple([state[i]] + list(state[interdep]))  # the value of i-th locus should be the first entry of the 'label'.
    return ctrb_map[i][label]

def fitness(state, ctrb_map, inter_mat, negative=False):
    """
    Given a state(: a tuple/array of length N), 
    Return fitness value followed by associated contributions.
    """
    state = np.array(state).flatten()
    N = state.size
    ctrbs = [_calculate_ith_contribution(state, ctrb_map, inter_mat, i) for i in range(N)]
    fitness_value = sum(ctrbs) / N  # averaged fitness value --> btw 0 ~ 1
    if negative:
        fitness_value = -fitness_value
    return ctrbs + fitness_value

def NK_landscape(N, K, negative=False):
    inter_mat = dependency(N, K)
    ctrb_map = contri(N, K)
    
    states = itertools.product(range(2), repeat=N)
    landscape_dic = {state: fitness(state, ctrb_map, inter_mat, negative=negative) for state in states}


