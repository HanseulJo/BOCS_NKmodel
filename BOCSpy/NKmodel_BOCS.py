import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from sample_models import sample_models
from NKmodel import NKmodel

N, K, A = 6, 1, 2
nkmodel = NKmodel(N, K, A=A)

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

# Save inputs in dictionary
inputs = {}
inputs['n_vars']     = N
inputs['evalBudget'] = 20
inputs['n_init']     = 2
inputs['lambda']     = 1e-4

# Save objective function and regularization term
inputs['model']    = evaluate # compute x^TQx row-wise
inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x,axis=1)

# Generate initial samples for statistical models
inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
inputs['y_vals']   = inputs['model'](inputs['x_vals'])

# Run BOCS-SA and BOCS-SDP (order 2)
(BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, 'SDP-l1')

BOCS_SA_obj = np.concatenate((inputs["y_vals"], BOCS_SA_obj))
BOCS_SDP_obj = np.concatenate((inputs["y_vals"], BOCS_SDP_obj))

# Compute optimal value found by BOCS
iter_t = np.arange(BOCS_SA_obj.size) + 1
BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)
BOCS_SDP_opt = np.minimum.accumulate(BOCS_SDP_obj)

# Compute minimum of objective function
n_models = 2**inputs['n_vars']
x_vals = np.zeros((n_models, inputs['n_vars']))
str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
for i in range(n_models):
	model = str_format.format(i)
	x_vals[i,:] = np.array([int(b) for b in model])
f_vals = inputs['model'](x_vals) #+ inputs['penalty'](x_vals)
opt_f  = np.min(f_vals)

fit_opt, states_opt, landscape = nkmodel.get_global_optimum(cache=True)
assert fit_opt == -opt_f

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

# -- END OF FILE --