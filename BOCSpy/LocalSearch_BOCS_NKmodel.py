import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from LinReg import LinReg
from NKmodel import NKmodel, generate_random_seeds_nkmodel
from BFLS import bfls_neighbor_suggestion
from MCTS import mcts_neighbor_suggestion

def main(kwargs):
    # Random Seed
    random_seeds = generate_random_seeds_nkmodel()
    im_seed_num_ = kwargs['interdependency_seed']
    im_seed_ = sorted(random_seeds.keys())[im_seed_num_]
    ctrbs_seed_list_, _ = sorted(random_seeds[im_seed_])
    ctrbs_seed_num_ = kwargs['payoff_seed']
    ctrbs_seed_ = ctrbs_seed_list_[ctrbs_seed_num_]

    # Create NK model
    N, K, A = kwargs['N'], kwargs['K'], kwargs['A']
    nkmodel = NKmodel(N, K, A=A, random_seeds=(im_seed_, ctrbs_seed_))

    # Helper functions : uses 'N' and 'nkmodel'
    def evaluate(x):
        """ True Objective function """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        assert x.shape[1] == N
        return np.concatenate([_evaluate_single(x[i]) for i in range(x.shape[0])], axis=0)
    def _evaluate_single(x):
        assert len(x) == N
        if x.ndim == 2:
            x = np.squeeze(x, axis=0)
        evaluation = nkmodel.fitness(tuple(x), negative=True)  # To solve minimization problem, "negative=True."
        return np.array([evaluation])  # 1 by 1 array
    def flip(state, ind):
        """ given a state (in numpy array), flip (0 <-> 1) a digit """
        assert state.ndim == 1
        new_state = state.copy()
        new_state[ind] = 1-new_state[ind]
        return new_state
    def neighbors(state):
        """ given a state (in numpy array), return a 2D array whose rows are neighbors of the state """
        return np.concatenate([flip(state, i).reshape(1,-1) for i in range(N)], axis=0)

    # Given parameters
    n_eval = kwargs['n_eval']
    n_init = kwargs['n_init']
    assert n_eval > N

    # Start with a random state
    x = np.zeros((n_init, N)).astype(int)
    y = np.zeros(n_init)
    init_x = np.random.choice(range(A), size=N).reshape(1,-1)  # random initial state
    x[0,:] = init_x
    y[0] = evaluate(init_x)

    # Produce more initial states, by randomly walking from init_x
    next_x = init_x.copy()
    for t in range(1, n_init):
        flip_ind = np.random.randint(N)
        next_x = flip(next_x, flip_ind)
        x[t,:] = next_x
        y[t] = evaluate(next_x)

    # input dictionary to pass data
    inputs = {
        'x_vals': x,
        'y_vals': y,
    }

    # Train initial statistical model
    LR = LinReg(nVars=N, order=2)
    LR.train(inputs)
        
    # BOCS
    for t in tqdm(range(n_init, n_eval+1)):
        # Setup statistical model objective for SA
        stat_model = lambda x: LR.surrogate_model(x, LR.alpha)

        # neighbor suggestion
        if t <= n_eval - 6:
            x_new, _ = bfls_neighbor_suggestion(stat_model, inputs, flips=n_eval-t+1)
        else:
            x_new, _ = mcts_neighbor_suggestion(stat_model, inputs, flips=n_eval-t+1)

		# evaluate model objective at new evaluation point
        x_new = x_new.reshape((1,-1))
        y_new = evaluate(x_new)

		# Update inputs dictionary
        inputs['x_vals'] = np.vstack((inputs['x_vals'], x_new))
        inputs['y_vals'] = np.hstack((inputs['y_vals'], y_new))

		# re-train linear model
        LR.train(inputs)


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Optimization of NK model with BOCS (SA / SDP)')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=20)
    parser_.add_argument('--n_init', dest='n_init', type=int, default=2)
    parser_.add_argument('--N', dest='N', type=int, default=6)
    parser_.add_argument('--K', dest='K', type=int, default=1)
    parser_.add_argument('--A', dest='A', type=int, default=2)
    parser_.add_argument('--interdependency_seed', dest='interdependency_seed', type=int, default=None)
    parser_.add_argument('--contribution_seed', dest='contribution_seed', type=int, default=None)
    #parser_.add_argument('--init_point_seed', dest='init_point_seed', type=int, default=None)
    parser_.add_argument('--start_from_bottom', dest='start_from_bottom', action='store_true', default=False)

    args_ = parser_.parse_args()
    kwargs_ = vars(args_)
    if args_.interdependency_seed is None:
        kwargs_['interdependency_seed'] = np.random.randint(100)
    if args_.payoff_seed is None:
        kwargs_['contribution_seed'] = np.random.randint(100)
    print(kwargs_)
    main(kwargs_)