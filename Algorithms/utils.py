import numpy as np
from itertools import product

def flip(state, ind):
    """ given a state (in numpy array), flip (0 <-> 1) a digit """
    assert state.ndim == 1, state.ndim
    new_state = state.copy()
    new_state[ind] = 1-new_state[ind]
    return new_state

def neighbors(state):
    """ given a state (in numpy array), return a 2D array whose rows are neighbors of the state """
    return np.stack([flip(state, i) for i in range(len(state))])

def hamming_dist(start, end):
    assert start.ndim == end.ndim == 1 , (start.ndim, end.ndim)
    assert start.size == end.size, (start.size, end.size)
    return (start != end).sum()

def is_reachable(start, end, flips):
    return (start + end).sum() % 2 == flips % 2 and hamming_dist(start,end) <= flips

def path(start, end, flips):
    assert is_reachable(start, end, flips)
    if flips == 0:
        return np.array([[]])
    N = start.size
    diff = start != end
    flip_inds = np.arange(N)[diff]
    if flips > diff.sum():
        assert (flips-diff.sum()) % 2 == 0
        n = (flips-diff.sum()) // 2
        flip_inds = np.append(flip_inds, np.tile(np.random.choice(N, n), 2))
    np.random.shuffle(flip_inds)
    return flip_inds

def wander(start, flips):
    assert start.ndim == 1 and flips % 2 == 0
    n = flips // 2
    flip_inds = np.arange(start.shape[0])
    np.random.shuffle(flip_inds)
    flip_inds = np.tile(flip_inds[:n], 2)
    return flip_inds

def is_visited(x_vals, x_new):
    assert x_new.ndim == 1, x_new.ndim
    return np.all(x_new == x_vals, axis=1).any()

def states(N):
    return np.stack([np.array(tup) for tup in product(range(2), repeat=N)])

# add new data
def add_data(inputs, x, y):
  assert len(y) == 1
  inputs['x_vals'] = np.vstack((inputs['x_vals'], x))
  inputs['y_vals'] = np.hstack((inputs['y_vals'], y))