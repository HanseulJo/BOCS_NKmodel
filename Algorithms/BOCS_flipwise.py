import sys
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from .utils import *
from .next_selection import *
from .LinReg import LinReg


surrogate_model_dict = {
    'BOCS':      LinReg(order=2),
    'PolyReg':   Pipeline([('poly', PolynomialFeatures(interaction_only=True)), ('linear', LinearRegression())]),
    'PolyLasso': Pipeline([('poly', PolynomialFeatures(interaction_only=True)), ('linear', Lasso())]),
    'SVR':       SVR(),
}


def BOCS_loc(args, inputs_, evaluate, surrogate_model_name, back_to_best=True, ascent_trials=10, pre_calc_acq=False, tqdm_on=False, progress_on=False):
    n_eval = args.n_eval
    n_init = args.n_init
    N = args.N
    all_states = states(N)

    # copy inputs (containing initial state)
    inputs = {k:v for k, v in inputs_.items()}
    n_0 = inputs['x_vals'].shape[0]

    # Produce more initial states, by randomly walking
    next_x = inputs['x_vals'][-1]
    for t in range(n_0, n_init):
        flip_ind = random_next(inputs['x_vals'], next_x)
        next_x = flip(next_x.reshape(-1), flip_ind)
        next_y = evaluate(next_x)
        add_data(inputs, next_x, next_y)
    
    # progress bar
    if progress_on:
        print("Local Search with BOCS")
        PROG_LEN = args.terminal_size
        ratio = (n_init-1)/ n_eval
        progressed_len = int(PROG_LEN * ratio)
        progress_bar = f'Flip {n_init-1}/{n_eval} ({100.*ratio:.1f}%)' + '|' + '#'*progressed_len + '-'*(PROG_LEN-progressed_len) + '| '
        progress_bar += f"Latest: {inputs['y_vals'][-1]:.4f}, SoFarBest: {(inputs['y_vals'].min()):.4f}"
        sys.stdout.write(progress_bar)
        sys.stdout.flush()

    # Train initial statistical model
    LR = surrogate_model_dict[surrogate_model_name]
    LR.fit(inputs['x_vals'], inputs['y_vals'])

    PATH = None
    for t in range(n_init, n_eval+1):
        # flip index suggestion and move a step
        if PATH is None:
            stat_model = lambda x: LR.predict(x.reshape(1,N))[0]
            if pre_calc_acq:
                acqs_ = LR.predict(all_states)
                flip_ind = neighbor_suggestion(args, stat_model, inputs, n_eval - t + 1, tqdm_on=tqdm_on, trials=ascent_trials, calculated_acqs=acqs_)
            else:
                flip_ind = neighbor_suggestion(args, stat_model, inputs, n_eval - t + 1, tqdm_on=tqdm_on, trials=ascent_trials)
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
        # train_surrogate model
        LR.fit(inputs['x_vals'], inputs['y_vals'])

        if back_to_best and (PATH is None) and t >= n_eval - N:  # N is maximum possible hamming distance
            PATH, objective_y = monitor_reachable_best(inputs['x_vals'], inputs['y_vals'], t, n_eval)

        if progress_on:
            if progress_bar is not None:
                sys.stdout.write('\b' * (len(progress_bar)))
            ratio = t / n_eval
            progressed_len = int(PROG_LEN * ratio)
            progress_bar = f'Flip {t}/{n_eval} ({100.*ratio:.1f}%)' + '|' + '#'*progressed_len + '-'*(PROG_LEN-progressed_len) + '| '
            progress_bar += f"Latest: {float(next_y):.4f}, SoFarBest: {(inputs['y_vals'].max()):.4f}"
            sys.stdout.write(progress_bar)
            sys.stdout.flush()
            if t == n_eval:
                print()
    result = inputs['y_vals'][-1]
    sofar_best = (inputs['y_vals']).max()
    return result, sofar_best, inputs


def neighbor_suggestion(args, stat_model, inputs, max_flips, trials=10, tqdm_on=False, calculated_acqs=None, return_scores=False):
    assert max_flips > 0

    x_vals = inputs['x_vals'].copy()
    x = x_vals[-1]
    N = args.N
    x_nbrs = neighbors(x)

    # visited penalty
    visit_weight = lambda fl: 1. - (fl/args.n_eval)**2 if fl>1 else 1.
    vw = visit_weight(max_flips)

    if max_flips == 1:
        if calculated_acqs is None:
            ascent_scores = [stat_model(x_nbrs[i].reshape((1,-1))) for i in range(N)]
        else:
            nbrs_to_bin = [x_nbrs[i].dot(1 << np.arange(N)[::-1]) for i in range(N)]  # change x_nbrs[i]=array([0,0,1,0,1,1]) into integer 11.
            ascent_scores = [calculated_acqs[b] for b in nbrs_to_bin]
    elif max_flips > 1:
        ascent_scores = []
        it = tqdm(range(N), desc=f"Stochastic Ascent from {x}") if tqdm_on else range(N)
        for i in it:
            if calculated_acqs is None:
                _asc = [stochastic_ascent(stat_model, np.vstack((x_vals, x_nbrs[i])), max_flips-1, visit_weight) for _ in range(trials)]
            else:
                _asc = [stochastic_ascent(stat_model, np.vstack((x_vals, x_nbrs[i])), max_flips-1, visit_weight, calculated_acqs=calculated_acqs) for _ in range(trials)]
            ascent_scores.append(sum(_asc)/trials)  # average
        for i in range(N):
            if is_visited(x_vals, x_nbrs[i]):
                ascent_scores[i] *= vw  # give weight by a num <= 1
    
    ascent_scores = np.array(ascent_scores)
    if return_scores:
        return ascent_scores

    best_nbr_ind = np.argmax(ascent_scores)
    return best_nbr_ind

