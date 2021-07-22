"""
Author: Hanseul Cho
Date: 2021.07.21 (last updated)
"""

import numpy as np
import sys
import itertools

class NKmodel(object):
    """
    NKmodel class. A single NK model.
    Huge thanks to https://github.com/elplatt/nkmodel.git
    
    <PARAM>
    N: the number of loci
    K: the number of the other dependent loci for each locus (0 <= K <= N-1)
    A: the number of states that each locus can have (e.g.: 2 for binary variables)

    <some attributes>
    self.interdependence : interdependency matrix in 2D boolean numpy array.
    self.contributions   : list of dict's - ith dict maps (tuple (x_i, x_i0, x_i1, ..., x_iK) of length K) |--> (contribution(=payoff) f_i(x))
    """
    def __init__(self, N, K, A=2, interdependence=None, contributions=None, random_seeds=(None, None)):
        assert 0 <= K <= N-1
        self.N = N
        self.K = K
        self.A = A
        self.loci = range(N)
        if interdependence is None:
            # randomly generated interdependence matrix
            self.interdependence = np.full((N,N), False)
            rng_state_dep = np.random.RandomState(seed=random_seeds[0])
            for i in self.loci:
                dependence = [i] + list(rng_state_dep.choice(list(set(self.loci) - set([i])), size=K, replace=False))
                self.interdependence[i][dependence] = True
        else:
            self.interdependence = interdependence
        if contributions is None:
            self.contributions = [{} for _ in self.loci]
            rng_state_ctrb = np.random.RandomState(seed=random_seeds[1])
            for i in range(N):
                for label in itertools.product(range(A), repeat=K+1):  # K+1 subcollection of loci values that effects the locus i
                    self.contributions[i][label] = float(rng_state_ctrb.random())  # float [0, 1)
        else:
            self.contributions = contributions

    def calculate_ith_contribution(self, state, i):
        assert i in self.loci
        assert type(state) == np.ndarray
        interdep = self.interdependence[i].copy()
        interdep[i] = False
        label = tuple([state[i]] + list(state[interdep]))  # the value of i-th locus should be the first entry of the 'label'.
        return self.contributions[i][label]

    def fitness_and_contributions(self, state, negative=False):
        """
        Given a state(: a tuple/string of length N), 
        Return fitness value and a list of contributions of each loci.
        """
        ctrbs = []
        if type(state) == str:
            state = np.array([int(state[i]) for i in range(self.N)])
        else:
            state = np.array(state)
        for i in self.loci:
            ctrbs.append(self.calculate_ith_contribution(state, i))
        fitness_value = sum(ctrbs) / self.N  # normalized(averaged) fitness value.
        if negative:
            fitness_value = -fitness_value
        return fitness_value, ctrbs
    
    def fitness(self, state, negative=False):
        f, _ = self.fitness_and_contributions(state, negative=negative)
        return f
    
    def landscape(self, negative = False):
        """
        Return a dictionary mapping each state to its fitness value. (Naive algorithm)
        """
        result_dic = {}
        for state in itertools.product(range(self.A), repeat=self.N): # along all possible states
            fitness, _= self.fitness_and_contributions(state)
            if negative:
                fitness = -fitness
            result_dic[state] = fitness
        return result_dic

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
    
    def print_info(self, path=None):
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
        else:
            with open(path + "/knowledge.txt", "w") if path is not None else sys.stdout as f1:
                for i in range(self.N):
                    print("".join(["X" if b else "O" for b in self.interdependence[i]]), file=f1)
            with open(path + "/landscape.txt", "w") if path is not None else sys.stdout as f2:
                d = self.landscape_with_contributions()
                for state, (fit, ctrbs) in d.items():
                    ctrbs = [str(round(v, 4)) for v in ctrbs]
                    fit = str(round(fit, 4))
                    state = "".join([str(x) for x in state])
                    print("\t".join([state] + ctrbs + [fit]), file=f2)
        order = min(10, 2**self.N)
        optlist = self.get_optimum_and_more(order)
        for i in range(order):
            opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
            print(f"{i+1}-th optimum: {opt} {optstates}")
    
    def rank_by_fitness(self, fitness_value, given_landscape=None):
        raise NotImplementedError