# Combinatorial Bayesian Optimization with restricted search steps: focusing on NK fitness landscape

## Dependencies (Packages)

```bash
cvxopt==1.2.6   # important to run codes related to 'bocs'
cvxpy           # important to run codes related to 'bocs'
easydict        # Not really necessary, but for clarity of codes.
matplotlib      # Not that necessary
numpy           # The most important.
pandas==1.3.0   # Not that necessary.
scikit-learn    # Quite important. Without this, you cannot use any surrogate model using simple regression AND BOCS.
scipy           # You will get this while installing numpy.
```

## Play a Game!

Type the following command into your personal terminal.

```bash
python Game.py
```

All the dependencies are in folders "Algorithms", "Models", and a text file "requirements.txt". 

If you open the file "Game.py" (with bunch of code lines written in Python), you will find:
```bash
GAME(args, chances=18, show_interdependence=False, surrogate_model_name='PolyReg', back_to_best=False, ascent_trial=64)
```
... at almost the last line of it. You can modify the input values as follows:
* `chances`: Positive Integer (>1). The number of flips that you are allowed to do.
* `surrogate_model_name`: String. The name of surrogate model, but you should use either 'PolyReg', 'PolyLasso', or 'BOCS'. Note that 'BOCS' is much slower(x3~4) than the others, so it is recommended to use 'PolyReg' or 'PolyLasso'.
* `back_to_best`: Boolean: Either True or False. If it is True, 'Back-to-reachable-best' heuristic will be applied at the last few steps. 
* `ascent_trial`: Positive Integer. The larger number you put here, the more accuracy you get; the more running time you take. 

If you want to manipulate more advanced features, you may modify the following part (line 251~257):
```bash
args.random_seed = 2021 # Any number!
args.n_eval = 18        # no need to change, if you already modified 'chances' above.
args.n_init = 2         # Do not change
args.N = 6              # 
args.K = 1              # You know, if you are already professional on NK model.
args.A = 2              # Binary variables: 2. Otherwise, you may put any integer greater than 2. 
args.terminal_size = 50 # Do not change
```


## Reference

Without the following wonderful works, my work would be impossible to be created.

```bash
https://github.com/baptistar/BOCS.git  # Original repository
https://github.com/elplatt/nkmodel.git
```