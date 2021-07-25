import subprocess

start_from_bottom = True

command = lambda x: f"python BOCSpy/NKmodel_BOCS.py --N 6 --K 1 --A 2 --n_eval 20 --n_init 2 --interdependency_seed {x[0]} --payoff_seed {x[1]} {'--start_from_bottom' if start_from_bottom else ''} "

error_count = 0
n_im = 10
n_ct = 10
for i in range(n_im):
    for j in range(n_ct):
        try:
            subprocess.run([command([i,j])], shell=True, check=True)
        except:
            error_count += 1

print(f"    {error_count}/{n_im*n_ct} Errors Occured.   ")
