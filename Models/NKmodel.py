"""
Author: Hanseul Cho
Date: 2021.08.04
"""

import numpy as np
import itertools

# Helper functions
def flip(state, ind):
    """ given a state (in numpy array), flip (0 <-> 1) a digit """
    assert state.ndim == 1, state.ndim
    new_state = state.copy()
    new_state[ind] = 1-new_state[ind]
    return new_state

def neighbors(state):
    """ given a state (in numpy array), return a 2D array whose rows are neighbors of the state """
    return np.stack([flip(state, i) for i in range(len(state))])

# Class object of NKmodel
class NKmodel(object):
    """
    NKmodel class. A single NK model.
    Huge thanks to https://github.com/elplatt/nkmodel.git
    
    <PARAM>
    N: the number of loci
    K: the number of the other dependent loci for each locus (0 <= K <= N-1)
    A: the number of states that each locus can have (e.g.: 2 for binary variables)
    <attributes>
    self.interdependence : interdependency matrix in 2D boolean numpy array.
    self.contributions   : list of dict's - ith dict maps (tuple (x_i, x_i0, x_i1, ..., x_iK) of length K) |--> (contribution(=payoff) f_i(x))
    """
    def __init__(self, N, K, A=2, interdependence=None, contributions=None, random_seeds=(None, None)):
        assert 0 <= K <= N-1
        self.N, self.K, self.A = N, K, A
        if interdependence is None:
            # randomly generated interdependence matrix
            self.interdependence = np.full((N,N), False)
            rng_state_dep = np.random.RandomState(seed=random_seeds[0])
            for i in range(N):
                dependence = [i] + list(rng_state_dep.choice(list(set(range(N)) - set([i])), size=K, replace=False))
                self.interdependence[i][dependence] = True
        else:
            self.interdependence = interdependence
        if contributions is None:
            self.contributions = [{} for _ in range(N)]
            rng_state_ctrb = np.random.RandomState(seed=random_seeds[1])
            for i in range(N):
                for label in itertools.product(range(A), repeat=K+1):  # K+1 subcollection of loci values that effects the locus i
                    self.contributions[i][label] = float(rng_state_ctrb.random())  # float [0, 1)
        else:
            self.contributions = contributions

    def _calculate_ith_contribution(self, state, i):
        assert i in range(self.N), i
        assert type(state) == np.ndarray, type(state)
        interdep = self.interdependence[i].copy()
        interdep[i] = False
        label = tuple([state[i]] + list(state[interdep]))  # the value of i-th locus should be the first entry of the 'label'.
        return self.contributions[i][label]

    def fitness_and_contributions(self, state, negative=False):
        """
        Given a state(: a tuple/string of length N), 
        Return fitness value and a list of contributions of each loci.
        """
        if type(state) == str:
            state = np.array([int(state[i]) for i in range(self.N)])
        else:
            state = np.array(state)
        ctrbs = [self._calculate_ith_contribution(state, i) for i in range(self.N)]
        fitness_value = sum(ctrbs) / self.N  # averaged fitness value --> btw 0 ~ 1
        if negative:
            fitness_value = -fitness_value
        return fitness_value, ctrbs
    
    def fitness(self, state, negative=False):
        f, _ = self.fitness_and_contributions(state, negative=negative)
        return f
    
    def evaluate(self, state, negative=False):
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        assert state.shape[1] == self.N, state.shape[1]
        return np.array([self.fitness(state[i], negative=negative) for i in range(state.shape[0])])
    
    def fitness_and_contrib_diff(self, state_prev, flip_ind, fitness_prev=None, ctrbs_prev=None, negative=False):
        state_new = flip(state_prev, flip_ind)
        depend_on_flip = self.interdependence[:,flip_ind]
        if fitness_prev is None or ctrbs_prev is None:
            fitness_prev, ctrbs_prev = self.fitness_and_contributions(state_prev, negative=negative)
        fitness_new = fitness_prev * self.N
        ctrbs_diff = np.zeros(self.N)
        for j in np.nonzero(depend_on_flip)[0]:
            ctrb_new_j = self._calculate_ith_contribution(state_new, j)
            ctrb_prev_j = ctrbs_prev[j]
            fitness_new = fitness_new - ctrb_prev_j + ctrb_new_j
            ctrbs_diff[j] = ctrb_new_j - ctrb_prev_j
        fitness_new /= self.N
        return fitness_new, ctrbs_diff

    def landscape(self, negative=False):
        """
        Return a dictionary mapping each state to its fitness value. (Naive algorithm)
        """
        landscape_dic = {}
        states = itertools.product(range(self.A), repeat=self.N)
        for state in states:
            landscape_dic[state] = self.fitness(state, negative=negative)
        return landscape_dic

    def landscape_with_contributions(self):
        """
        Return a dictionary mapping each state to its fitness value and contributions of loci. (Naive algorithm)
        """
        return {state: self.fitness_and_contributions(state) for state in itertools.product(range(self.A), repeat=self.N)}  # along all possible states

    def get_global_optimum(self, negative=False, anti_opt=False, cache=False, given_landscape=None):
        """
        Global maximum fitness value and its maximizer(state), in a NAIVE way.
        If "anti_opt=True", this returns the "minimum" fitness value and "minimizer". 
        """
        landscape = self.landscape() if given_landscape is None else given_landscape
        optimum = max(landscape.values()) if not anti_opt else min(landscape.values())
        states = [s for s in landscape.keys() if landscape[s] == optimum]
        if negative:
            optimum = -optimum
        if cache:
            return optimum, states, landscape
        else:
            return optimum, states
        
    def get_optimum_and_more(self, order, negative=False, anti_opt=False, cache=False, given_landscape=None):
        """
        First several maximum fitness values and their maximizers(states), in a NAIVE way.
        If "anti_opt=True", this returns first several "minimum" fitness values and "minimizers". 
        """
        landscape = self.landscape() if given_landscape is None else given_landscape
        landscape_list = sorted(landscape.items(), key=lambda x: -x[1])
        if anti_opt:
            landscape_list.reverse()
        state_opt, fit_opt = landscape_list[0]
        if negative:
            fit_opt = -fit_opt
        optima2states = [{"fitness": fit_opt, "states":[state_opt]}]
        cnt = 1
        for state, fitness in landscape_list[1:]:
            if negative:
                fitness = -fitness
            if fitness == optima2states[-1]["fitness"]:
                optima2states[-1]["states"].append(state)
            else:
                cnt += 1
                if cnt > order:
                    break
                optima2states.append({"fitness": fitness, "states":[state]})
        if cache:
            return optima2states, landscape
        else:
            return optima2states
    
    def print_info(self, path=None, order=10):
        optlist = self.get_optimum_and_more(order)
        if path is None:
            print("\nInterdependence Matrix:")
            for i in range(self.N):
                print("".join(["X" if b else "O" for b in self.interdependence[i]]))
            print("\nLandscape:")
            d = self.landscape_with_contributions()
            for state, (fit, ctrbs) in d.items():
                ctrbs = [str(round(v, 4)) for v in ctrbs]
                fit = str(round(fit, 4))
                state = "".join([str(x) for x in state])
                print("\t".join([state] + ctrbs + [fit]))
            for i in range(order):
                opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
                print(f"{i+1}-th optimum: {opt} {optstates}")
        else:
            with open(path + "/knowledge.txt", "w") as f1:
                for i in range(self.N):
                    print("".join(["X" if b else "O" for b in self.interdependence[i]]), file=f1)
            with open(path + "/landscape.txt", "w") as f2:
                d = self.landscape_with_contributions()
                for state, (fit, ctrbs) in d.items():
                    ctrbs = [str(round(v, 4)) for v in ctrbs]
                    fit = str(round(fit, 4))
                    state = "".join([str(x) for x in state])
                    print("\t".join([state] + ctrbs + [fit]), file=f2)
            with open(path + "/rankboard.txt", "w") as f3:
                for i in range(order):
                    opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
                    print(f"{i+1}-th optimum: {opt} {optstates}", file=f3)


# Fuctions to generate random seeds for NKmodel
def _generate_random_seeds(seed_str, n_im_seed=3, n_ctrbs_seed=3, n_init_point_seed=3):
    """
    Original code: COMBO.experiments.random_seed_config.py
    """
    rng_state = np.random.RandomState(seed=sum([ord(ch) for ch in seed_str]))
    result = {}
    for _ in range(n_im_seed):
        result[rng_state.randint(0, 10000)] = (list(rng_state.randint(0, 10000, (n_ctrbs_seed,))), list(rng_state.randint(0, 10000, (n_init_point_seed,))))
    return result

def generate_random_seeds_nkmodel():
    """
    Original code: COMBO.experiments.random_seed_config.py
    """
    return _generate_random_seeds(seed_str="NK_MODEL", n_im_seed=100, n_ctrbs_seed=100, n_init_point_seed=100)


# Easy Construction of NKmodel
def Construct_NKmodel(kwargs, im_seed_num=None, ctrbs_seed_num=None, verbose=False):

    if im_seed_num is None:
        im_seed_num = np.random.randint(100)
    if ctrbs_seed_num is None:
        ctrbs_seed_num = np.random.randint(100)
    
    # Random Seed for NKmodel
    random_seeds = generate_random_seeds_nkmodel()
    im_seed_ = sorted(random_seeds.keys())[im_seed_num]
    ctrbs_seed_list_, _ = sorted(random_seeds[im_seed_])
    ctrbs_seed_ = ctrbs_seed_list_[ctrbs_seed_num]

    # Create NK model
    nkmodel = NKmodel(kwargs['N'], kwargs['K'], A=kwargs['A'], random_seeds=(im_seed_, ctrbs_seed_))

    # Miscelleneous
    if verbose:
        print(f"im_seed_num {im_seed_num} ctrbs_seed_num {ctrbs_seed_num}")

    return nkmodel