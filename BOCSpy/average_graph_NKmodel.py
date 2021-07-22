import os, pickle, glob
import numpy as np
import matplotlib.pyplot as plt

save_dir = os.path.join("BOCSpy", "NKmodel_BOCS_Runs")
im_seed = ''  # ''

seed_str = f"seed{im_seed}R" if im_seed else ''
filenames = glob.glob(os.path.join(save_dir, f'inputs_{seed_str}*.pickle'))
bo_datas = [pickle.load(open(fn, 'rb')) for fn in filenames]
print(f"Using {len(filenames)} runs to make a plot...")

keys = ['BOCS_SA_opt', 'BOCS_SDP_opt', 'opt_f']
for i in range(len(bo_datas))[::-1]:
    if not all([k in bo_datas[i] for k in keys]):
        del bo_datas[i]

optimums = np.array([data['opt_f'] for data in bo_datas]).reshape(-1,1)
BOCS_SA_opts = np.concatenate([data['BOCS_SA_opt'].reshape(1,-1) for data in bo_datas])
BOCS_SDP_opts = np.concatenate([data['BOCS_SDP_opt'].reshape(1,-1) for data in bo_datas])
BOCS_SA_datas = BOCS_SA_opts - optimums
BOCS_SDP_datas = BOCS_SDP_opts - optimums

length = len(BOCS_SA_datas[0])
BOCS_SA_means = BOCS_SA_datas.mean(axis=0)
BOCS_SA_stds = BOCS_SA_datas.std(axis=0)
BOCS_SDP_means = BOCS_SDP_datas.mean(axis=0)
BOCS_SDP_stds = BOCS_SDP_datas.std(axis=0)
x = np.arange(1,length+1)
print("BOCS_SA result (gap):", BOCS_SA_means[-1])
print("BOCS_SDP result (gap):", BOCS_SDP_means[-1])

plt.plot(x, BOCS_SA_means, linewidth=2, label='BOCS_SA', color='r')
plt.fill_between(x, (BOCS_SA_means - BOCS_SA_stds), (BOCS_SA_means + BOCS_SA_stds), color='r', alpha=0.2)
plt.plot(x, BOCS_SDP_means, linewidth=2, label='BOCS_SDP', color='b')
plt.fill_between(x, (BOCS_SDP_means - BOCS_SDP_stds), (BOCS_SDP_means + BOCS_SDP_stds), color='b', alpha=0.2)
plt.title('Gap: Global Optimum - Tentative Optima (Mean $\pm$ Std.dev)')
plt.legend()
plt.xticks(range(0, 21, 4))
plt.ylim(-0.05, 0.55)
plt.show()



