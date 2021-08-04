import sys, time

from .utils import *
from .next_selection import *


def Random_Walk(args, inputs_, evaluate, back_to_best=True, progress_on=False):
    if progress_on:
        print("Random Walk")
        PROG_LEN = args.terminal_size 
        progress_bar = None
    n_eval = args.n_eval
    N = args.N

    # copy inputs (containing initial state)
    inputs = {k:v for k, v in inputs_.items()}
    n_init = inputs['x_vals'].shape[0]

    # randomly walk
    next_x = inputs['x_vals'][-1].copy()
    PATH = None
    for t in range(n_init, n_eval+1):
        # flip index suggestion and move a step
        if PATH is None:
            flip_ind = random_next(inputs['x_vals'], next_x)
            next_x = flip(next_x.reshape(-1), flip_ind)
            next_y = evaluate(next_x)
            add_data(inputs, next_x, next_y)
        elif back_to_best:
            left_chance = n_eval - t + 1
            flip_ind = PATH[-left_chance]
            next_x = flip(next_x.reshape(-1), flip_ind)
            next_y = evaluate(next_x)
            add_data(inputs, next_x, next_y)
            if left_chance-1>0 and (left_chance-1) % 2 == 0 and next_y > objective_y:  # new reachable best --> update PATH
                PATH[-(left_chance-1):] = wander(next_x, left_chance-1)

        if back_to_best and (PATH is None) and t >= n_eval - N:  # N is maximum possible hamming distance
            PATH, objective_y = monitor_reachable_best(inputs['x_vals'], inputs['y_vals'], t, n_eval)
        
        if progress_on:
            if progress_bar is not None:
                sys.stdout.write('\b' * (len(progress_bar)+1))
            ratio = t / n_eval
            progressed_len = int(PROG_LEN * ratio)
            progress_bar = f'Flip {t}/{n_eval} ({100.*ratio:.1f}%)' + '|' + '#'*progressed_len + '-'*(PROG_LEN-progressed_len) + '| '
            progress_bar += f"Latest: {inputs['y_vals'][-1]:.4f}, SoFarBest: {(inputs['y_vals'].max()):.4f}"
            sys.stdout.write(progress_bar)
            sys.stdout.flush()
            if t == n_eval:
                print()
    result = inputs['y_vals'][-1]
    sofar_best = (inputs['y_vals']).max()
    return result, sofar_best, inputs
