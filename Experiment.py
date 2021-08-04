import random, time
import matplotlib.pyplot as plt
from easydict import EasyDict
from Models.NKmodel import *
from Algorithms.utils import *
from Algorithms.next_selection import random_next
from Algorithms.random_search import Random_Walk
from Algorithms.BOCS_flipwise import BOCS_loc

def RunV2(args, im_seed_num=None, ctrbs_seed_num=None, start_from_bottom=False, show_seed=True, show_landscape=False, progress_on=True):

    nkmodel = Construct_NKmodel(args, im_seed_num=im_seed_num, ctrbs_seed_num=ctrbs_seed_num, verbose=show_seed)
    max_fit, _, _landscape = nkmodel.get_global_optimum(cache=True)
    min_fit, _ = nkmodel.get_global_optimum(anti_opt=True, given_landscape=_landscape)
    range_fit = max(max_fit - min_fit, 1e-8)
    if show_landscape:
        nkmodel.print_info()
    def evaluate(x):
        return nkmodel.evaluate(x)  # Minimization Problem: negative=True

    if start_from_bottom:
        _, anti_opt_state = nkmodel.get_global_optimum(anti_opt=True)
        _state_tuple = random.choice(anti_opt_state)
        init_x = np.array(_state_tuple)
    else:
        init_x = np.random.choice(range(args.A), size=args.N)  
    init_y = evaluate(init_x)
    inputs = {'x_vals': init_x.reshape((1,-1)), 'y_vals': init_y}

    # Produce more initial states, by randomly walking from init_x
    next_x = init_x.copy()
    for t in range(1, args.n_init):
        flip_ind = random_next(inputs['x_vals'], next_x)
        next_x = flip(next_x.reshape(-1), flip_ind)
        next_y = evaluate(next_x)
        add_data(inputs, next_x, next_y)
    print(inputs)

    optlist = nkmodel.get_optimum_and_more(64, given_landscape=_landscape)
    r_best = None
    for i in range(64):
        opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
        if r_best is None and any([is_reachable(init_x, np.array(st), args.n_eval) for st in optstates]):
            r_best = opt 
            break

    print("Comparison started")
    Comparison = EasyDict({})

    rw = EasyDict({})
    t = time.time()
    f_, b_, inp = Random_Walk(args, inputs, evaluate, progress_on=progress_on, back_to_best=False)
    print(f"Random Walk Naive runtime: {time.time()-t:.4f}sec")
    rw.loss, rw.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.randomwalk = rw
    print()

    rw_btrb = EasyDict({})
    t = time.time()
    f_, b_, inp = Random_Walk(args, inputs, evaluate, progress_on=progress_on, back_to_best=True)
    print(f"Random Walk w/ BTRB runtime: {time.time()-t:.4f}sec")
    rw_btrb.loss, rw_btrb.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.randomwalk_btrb = rw_btrb
    print()

    bocs = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='BOCS', progress_on=progress_on, back_to_best=False, ascent_trials=10, pre_calc_acq=True)
    print(f"BOCS_LOC Naive (ascent trial 10) runtime: {time.time()-t:.4f}sec")
    bocs.loss, bocs.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.bocs = bocs
    print()
    
    bocs_btrb = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='BOCS', progress_on=progress_on, back_to_best=True, ascent_trials=10, pre_calc_acq=True)
    print(f"BOCS_LOC w/ BTRB (ascent trial 10) runtime: {time.time()-t:.4f}sec")
    bocs_btrb.loss, bocs_btrb.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.bocs_btrb = bocs_btrb
    print()

    polyreg = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='PolyReg', progress_on=progress_on, back_to_best=False, ascent_trials=10, pre_calc_acq=True)
    print(f"POLY_REG Naive (ascent trial 10) runtime: {time.time()-t:.4f}sec")
    polyreg.loss, polyreg.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.polyreg = polyreg
    print()

    polyreg_btrb = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='PolyReg', progress_on=progress_on, back_to_best=True, ascent_trials=10, pre_calc_acq=True)
    print(f"POLY_REG w/ BTRB (ascent trial 10) runtime: {time.time()-t:.4f}sec")
    polyreg_btrb.loss, polyreg_btrb.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.polyreg_btrb = polyreg_btrb
    print()

    lasso = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='PolyLasso', progress_on=progress_on, back_to_best=False, ascent_trials=10, pre_calc_acq=True)
    print(f"POLY_LASSO Naive (ascent trial 10) runtime: {time.time()-t:.4f}sec")
    lasso.loss, lasso.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.lasso = lasso
    print()

    lasso_btrb = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='PolyLasso', progress_on=progress_on, back_to_best=True, ascent_trials=10, pre_calc_acq=True)
    print(f"POLY_LASSO w/ BTRB (ascent trial 10) runtime: {time.time()-t:.4f}sec")
    lasso_btrb.loss, lasso_btrb.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.lasso_btrb = lasso_btrb
    print()

    temp_n_init, args.n_init = args.n_init, args.n_eval - args.N
    print("args.n_init became", args.n_init)
    print()

    bocs_eff = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='BOCS', progress_on=progress_on, back_to_best=False, ascent_trials=64, pre_calc_acq=True)
    print(f"BOCS_LOC Efficient (ascent trial 64) runtime: {time.time()-t:.4f}sec")
    bocs_eff.loss, bocs_eff.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.bocs_eff = bocs_eff
    print()

    polyreg_eff = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='PolyReg', progress_on=progress_on, back_to_best=False, ascent_trials=64, pre_calc_acq=True)
    print(f"POLY_REG Efficient (ascent trial 64) runtime: {time.time()-t:.4f}sec")
    polyreg_eff.loss, polyreg_eff.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.polyreg_eff = polyreg_eff
    print()

    lasso_eff = EasyDict({})
    t = time.time()
    f_, b_, inp = BOCS_loc(args, inputs, evaluate, surrogate_model_name='PolyLasso', progress_on=progress_on, back_to_best=False, ascent_trials=64, pre_calc_acq=True)
    print(f"POLY_LASSO Efficient (ascent trial 64) runtime: {time.time()-t:.4f}sec")
    lasso_eff.loss, lasso_eff.ys = (r_best - f_)/range_fit, (inp['y_vals']-min_fit)/range_fit
    Comparison.lasso_eff = lasso_eff
    print()
    
    args.n_init = temp_n_init
    print("args.n_init became", args.n_init)
    print()

    for i in range(10):
        opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
        print(f"{i+1}-th optimum: {opt:.8f} {optstates}")
    print()

    return Comparison, r_best, optlist[0]["fitness"]

if __name__ == '__main__':
    args = EasyDict()

    args.random_seed = 2021
    args.n_eval = 18
    args.n_init = 2
    args.N = 6
    args.K = 1
    args.A = 2
    args.terminal_size = 50

    np.set_printoptions(precision=3)

    #random.seed(args.random_seed)
    #np.random.seed(args.random_seed)

    compare_all_sfb = EasyDict({
        'randomwalk': {'loss':[], 'ys':[]},
        'randomwalk_btrb': {'loss':[], 'ys':[]},
        'bocs': {'loss':[], 'ys':[]},
        'bocs_btrb': {'loss':[], 'ys':[]},
        'bocs_eff': {'loss':[], 'ys':[]},
        'polyreg': {'loss':[], 'ys':[]},
        'polyreg_btrb': {'loss':[], 'ys':[]},
        'polyreg_eff': {'loss':[], 'ys':[]},
        'lasso': {'loss':[], 'ys':[]},
        'lasso_btrb': {'loss':[], 'ys':[]},
        'lasso_eff': {'loss':[], 'ys':[]},
    })

    iter = [(im, ctrbs, init_) for im in range(10) for ctrbs in range(10) for init_ in range(1)]
    resume = 0
    for i in range(resume, len(iter)):
        im, ctrbs, _ = iter[i]
        try:
            comparison, r_best, glob_opt = RunV2(args, im_seed_num=im, ctrbs_seed_num=ctrbs, start_from_bottom=True)
        except KeyboardInterrupt:
            args.n_init = 2
            raise KeyboardInterrupt(f'stopped at iter[{i}]')
        except TimeoutError:
            args.n_init = 2
            continue
        except Exception:
            args.n_init = 2
            raise Exception(f'stopped at iter[{i}]')
        for k in comparison.keys():
            compare_all_sfb[k].loss.append(comparison[k].loss)
            compare_all_sfb[k].ys.append(comparison[k].ys)
    
    for k in compare_all_sfb.keys():
        loss_arr = np.array(compare_all_sfb[k].loss)
        print(f"{k:<12} ; Loss=(r_best-final)/(maxfit-minfit) = {loss_arr.mean():.5f}, std {loss_arr.std():.5f}")
    print()
    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.ylim(-0.05, 1)
    plt.rc('font', size=15)
    for k in compare_all_sfb.keys():
        averaged_ys = np.stack(compare_all_sfb[k].ys).mean(axis=0)
        print(f"{k:<12} ; AVG graph (ys):", averaged_ys)
        plt.plot(np.arange(19), averaged_ys, label=k)
    plt.xticks(ticks=np.arange(19))
    plt.yticks(ticks=np.arange(10)/10)
    plt.legend()
    plt.plot()