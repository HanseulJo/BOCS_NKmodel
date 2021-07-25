import os, pickle, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from sample_models import sample_models
from NKmodel import NKmodel, generate_random_seeds_nkmodel


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

    # Save inputs in dictionary
    inputs = {}
    inputs['n_vars']     = N
    inputs['evalBudget'] = kwargs['n_eval']
    inputs['n_init']     = kwargs['n_init']
    inputs['lambda']     = kwargs['lambda']
    inputs['NKmodel']    = nkmodel

    # Save results (1)
    save_dir = os.path.join("BOCSpy", "NKmodel_BOCS_Runs")
    seed_str = f"seed{im_seed_:04d}R{ctrbs_seed_:04d}"

    # Save objective function and regularization term
    def evaluate(x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        assert x.shape[1] == N
        return np.concatenate([_evaluate_single(x[i]) for i in range(x.shape[0])], axis=0)
    def _evaluate_single(x):
        #assert x.dim() == 1
        assert len(x) == N
        if x.ndim == 2:
            x = np.squeeze(x, axis=0)
        evaluation = nkmodel.fitness(tuple(x), negative=True)  # To solve minimization problem, "negative=True."
        return np.array([evaluation])  # 1 by 1 array

    inputs['model']    = evaluate
    inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x,axis=1)

    # Generate initial samples for statistical models
    if kwargs['start_from_bottom']:
        anti_opt_list = nkmodel.get_optimum_and_more(kwargs['n_init'], anti_opt=True)
        anti_opt_states, ind = [], 0
        while len(anti_opt_states) < kwargs['n_init']:
            anti_opt_states += anti_opt_list[ind]["states"]
            ind += 1
        anti_opt_states = anti_opt_states[:kwargs['n_init']]
        inputs['x_vals'] = np.concatenate([np.array(state).reshape(1,-1) for state in anti_opt_states])
    else:
        inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
    inputs['y_vals']   = inputs['model'](inputs['x_vals'])
    
    # Run BOCS-SA and BOCS-SDP (order 2)
    try:
        (BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')
        (BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, 'SDP-l1')
    except:
        # Error Handling
        import time
        error_log_dir = os.path.join(save_dir, "error_log", time.strftime('%y%m%d-%X', time.localtime(time.time())))
        os.makedirs(error_log_dir)
        nkmodel.print_info(path=error_log_dir)
        file_name = f"inputs_{seed_str}"
        del inputs['model']
        del inputs['penalty']
        with open(os.path.join(error_log_dir, file_name+".pickle"), 'wb') as f:
            pickle.dump(inputs, f, pickle.HIGHEST_PROTOCOL)
        del inputs['NKmodel']
        with open(os.path.join(error_log_dir, file_name+".txt"), 'w') as f:
            print(inputs, file=f)
        time.sleep(1.)
        raise RuntimeError
    
    BOCS_SA_obj = np.concatenate((inputs["y_vals"], BOCS_SA_obj))
    BOCS_SDP_obj = np.concatenate((inputs["y_vals"], BOCS_SDP_obj))

    # Compute optimal value found by BOCS
    iter_t = np.arange(BOCS_SA_obj.size) + 1
    BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)
    BOCS_SDP_opt = np.minimum.accumulate(BOCS_SDP_obj)
    inputs['BOCS_SA_opt'] = BOCS_SA_opt
    inputs['BOCS_SDP_opt'] = BOCS_SDP_opt

    # Compute minimum of objective function
    n_models = 2**inputs['n_vars']
    x_vals = np.zeros((n_models, inputs['n_vars']))
    str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
    for i in range(n_models):
        model = str_format.format(i)
        x_vals[i,:] = np.array([int(b) for b in model])
    f_vals = inputs['model'](x_vals) + inputs['penalty'](x_vals)
    opt_f  = np.min(f_vals)
    inputs['opt_f'] = opt_f

    fit_opt, _ = nkmodel.get_global_optimum()
    assert fit_opt == -opt_f # regularization = 0

    # Save results (2)
    runs = glob.glob(os.path.join(save_dir, f"inputs_{seed_str}_*.pickle"))
    runs = [int(os.path.basename(p).split(".")[0][-3:]) for p in runs]
    run_num = max(runs) + 1 if runs else 0
    file_name = f"inputs_{seed_str}_{run_num:03d}.pickle"
    del inputs['model']
    del inputs['penalty']
    with open(os.path.join(save_dir, file_name), 'wb') as f:
        pickle.dump(inputs, f, pickle.HIGHEST_PROTOCOL)

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(iter_t, np.abs(BOCS_SA_opt), color='r', label='BOCS-SA')
    ax.plot(iter_t, np.abs(BOCS_SDP_opt), color='b', label='BOCS-SDP')
    ax.axhline(y = fit_opt, color='tab:orange', linestyle='--', label='Global optimum')
    ax.set_xlabel('$t$')
    ax.set_ylabel('Best $f(x)$')
    ax.set_xticks(iter_t)
    ax.legend()
    fig.savefig('BOCS_NKmodel.pdf')
    plt.close(fig)

if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Optimization of NK model with BOCS (SA / SDP)')
    parser_.add_argument('--n_eval', dest='n_eval', type=int, default=20)
    parser_.add_argument('--n_init', dest='n_init', type=int, default=2)
    parser_.add_argument('--N', dest='N', type=int, default=6)
    parser_.add_argument('--K', dest='K', type=int, default=1)
    parser_.add_argument('--A', dest='A', type=int, default=2)
    parser_.add_argument('--interdependency_seed', dest='interdependency_seed', type=int, default=None)
    parser_.add_argument('--payoff_seed', dest='payoff_seed', type=int, default=None)
    #parser_.add_argument('--init_point_seed', dest='init_point_seed', type=int, default=None)
    parser_.add_argument('--start_from_bottom', dest='start_from_bottom', action='store_true', default=False)
    parser_.add_argument('--lambda', dest='lambda', type=float, default=0.)

    args_ = parser_.parse_args()
    kwargs_ = vars(args_)
    if args_.interdependency_seed is None:
        kwargs_['interdependency_seed'] = np.random.randint(100)
    if args_.payoff_seed is None:
        kwargs_['payoff_seed'] = np.random.randint(100)
    print(kwargs_)
    main(kwargs_)