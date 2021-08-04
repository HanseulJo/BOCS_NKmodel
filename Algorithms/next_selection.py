import numpy as np
from scipy.special import softmax

from .utils import *


def random_next(x_vals, curr_x=None):
    """
    RANDOM NEXT index to flip, but avoid visited nbh as much as possible
    """
    if curr_x is None:
        curr_x = x_vals[-1]
    num_nbh = curr_x.shape[0]  # N
    nbh_inds = np.arange(num_nbh)
    np.random.shuffle(nbh_inds)
    i_ = 0
    while i_ < num_nbh:
        flip_ind = nbh_inds[i_]
        if not is_visited(x_vals, flip(curr_x.reshape(-1), flip_ind)):
            break
        else:
            i_ += 1
    if i_ == num_nbh:
        flip_ind = np.random.randint(num_nbh)
    return flip_ind
    

def monitor_reachable_best(x_vals, y_vals, t, n_eval, show_objective_x=False):
    """
    Return a PATH to Reachable-best (and the value of R-best)
    """
    curr_x = x_vals[-1]
    left_chance = n_eval - t
    y_sort_ind = np.argsort(-y_vals)  # positive: min -y ~= max fitness
    PATH = None
    for j in y_sort_ind:
        objective_x = x_vals[j]
        objective_y = y_vals[j]
        if is_reachable(curr_x, objective_x, left_chance):
            PATH = path(curr_x, objective_x, left_chance)
            break
    if show_objective_x:
        return PATH, objective_y, objective_x
    return PATH, objective_y


def stochastic_ascent(stat_model, x_vals, max_flips, visit_weight, calculated_acqs=None):
    """
    Stochastic Ascent of Acquisition.
    """
    assert max_flips > 0
    x_vals = x_vals.copy()
    x = x_vals[-1]
    N = x.shape[0]

    for flip_ in range(1,max_flips+1):
        x_nbrs = neighbors(x)
        
        # evaluate acquisitions
        if calculated_acqs is None:
            nbrs_acquisition = np.array([stat_model(x_nbrs[i].reshape((1,-1))) for i in range(N)])
        else:
            nbrs_to_bin = [x_nbrs[i].dot(1 << np.arange(N)[::-1]) for i in range(N)]
            nbrs_acquisition = np.array([calculated_acqs[b] for b in nbrs_to_bin])

        # convert to probability (less acq, more prob)
        m_ = nbrs_acquisition.min()
        score = nbrs_acquisition.copy()
        if m_ < 0:
            score -= m_ # make all scores non negative
        vw = visit_weight(max_flips+1-flip_)
        for i in range(N):
            if is_visited(x_vals, x_nbrs[i]):
                score[i] *= vw  # give weight by a num <= 1 to visited vertex
        score_stand = softmax(score*2)  # multiplying 3: just for appropriate scaling
        #print(score_stand)

        # update x
        next_ind = int(np.random.choice(np.arange(N), p=score_stand))
        x = x_nbrs[next_ind]
        x_vals = np.concatenate((x_vals, x.reshape((1,-1))))

    return nbrs_acquisition[next_ind]
