import time
import numpy as np
from easydict import EasyDict

from Models.NKmodel import Construct_NKmodel
from Algorithms.utils import *
from Algorithms.next_selection import monitor_reachable_best
from Algorithms.BOCS_flipwise import surrogate_model_dict, neighbor_suggestion

def _show_best(best):
    print("*** Displaying Your Best Attempt So Far ***")
    for key in best:
        print("best", key, ":", best[key])

def _show_all(inputs):
    print("*** Displaying All Your Attempt So Far (state --> score) ***")
    x_vals, y_vals = inputs['x_vals'], inputs['y_vals']
    for i in range(y_vals.size):
        print(f"Round{i:02d}: {x_vals[i]} --> {y_vals[i]:.4f}")

def _yes_or_no():
    """
    example:
    if _yes_or_no():
        pass
    """
    YORN = 'undefined'
    while True:
        YORN = input("yes[y] or no[n]: ").strip()
        if len(YORN) <= 0:
            continue
        if YORN.lower()[0] in ['y', 'n']:
            break
    return YORN.lower()[0] == 'y'


def input_start_point(N):
    print(f"\nNK MODEL GAME MODE ON\n")
    print(f"Before start, write {N} numbers: each number should be 0 or 1.")
    print(f"Separate numbers by ' '(spacebar).")
    INPUT = []
    error_str=None
    while len(INPUT) != N or not set(INPUT).issubset(set(['0', '1'])):
        INPUT = input("* Your input: ").strip().split()
        if len(INPUT) != N:
            print(f"Wrong length: write {N} numbers and separate with spacebars.")
        if not set(INPUT).issubset(set(['0', '1'])):
            print("Wrong number: write 0 or 1 only")
    init_x = np.array([int(x) for x in INPUT])
    return init_x


def player_decision(N, inputs, best, next_x):
    print( "NOW: Which digit do YOU want to flip?")
    print(f"     Write a number m if you want to flip m-th digit: (1 <= m <= {N})")
    print(f"     Or, write 'BEST' if you want to display the BEST results so far")
    print(f"     Or, write 'history' if you want to display the ALL your attempts so far")
    while True:
        flip_idx = None
        while flip_idx not in range(1, N+1):
            INPUT = input("* Your input: ").strip()
            if INPUT == 'BEST':
                _show_best(best)
            elif INPUT == 'history':
                _show_all(inputs)
            elif INPUT.isdigit():
                flip_idx = int(INPUT)
                if flip_idx not in range(1, N+1):
                    print(f"Wrong number: Write a number from 1 to {N}: ")
            else:
                print("Wrong input format: Write again: ")
        flip_idx -= 1  # index: 0 ~ N-1
        print("Do you really want to change the state as follows?:")
        print(">> FROM :", next_x)
        print(">>   TO :", flip(next_x, flip_idx))
        if _yes_or_no():
            return flip_idx
        else:
            print("You answered NO: Re-write your input. ")
            print( "NOW: Which digit do U wanna flip?")
            print(f"     Write a number m if you want to flip m-th digit: (1 <= m <= {N})")
            print(f"     Or, Write 'BEST' if you wnat to display the BEST results so far")


def GAME(args, chances=None, show_interdependence=False, can_restart=False, surrogate_model_name='PolyReg', back_to_best=False, ascent_trial=64):

    print("Preparing the game.....")

    N = args.N
    if chances is None:
        chances = args.n_eval
    elif chances != args.n_eval:
        print("args.n_eval changed:", chances)
        args.n_eval = chances
    all_states = states(N)
    
    # Construct NKmodel
    nkmodel = Construct_NKmodel(args)
    optimum, _, _landscape = nkmodel.get_global_optimum(cache=True)
    optlist = nkmodel.get_optimum_and_more(2**N)
    if show_interdependence:
        print(nkmodel.interdependence) # Boolean interdependence matrix

    start = True
    while start:
        init_x = input_start_point(N)
        init_y = nkmodel.evaluate(init_x)
        inputs = {'x_vals': init_x.reshape(1,-1), 'y_vals': init_y}
        
        best = {'x': init_x.copy(), 'y': float(init_y), 'round': 1}  # store so far best result
        
        # Surrogate model
        LR = surrogate_model_dict[surrogate_model_name]
        
        fit_diff = 0                # fitness  difference (previous --> current)       
        ctrbs_diff = np.zeros(N)    # contrib. difference (previous --> current)

        next_x = init_x.copy()
        next_y = init_y
        PATH = None
        print()
        for ROUND in range(1, chances+1):
            print(f"*** Round {ROUND}/{chances} ***")
            print( "Previous results:")
            print( "Current state:", next_x, '<== IMPROVED' if ROUND>1 and np.all(best['x'] == next_x) else '')
            print(f"Current fitness: {float(next_y):.4f} ({fit_diff:.4f} from before)")
            print( "Current improvement of contributions:", ctrbs_diff)
            print()

            # Flip-index suggestion
            left_chance = chances - ROUND + 1
            if PATH is None:
                if ROUND > 1:
                    stat_model = lambda x: LR.predict(x.reshape(1,N))[0]
                    acqs_ = LR.predict(all_states)
                    ascent_score = neighbor_suggestion(args, stat_model, inputs, left_chance, trials=ascent_trial, tqdm_on=True, calculated_acqs=acqs_, return_scores=True)
                    flip_suggest = ascent_score.argmax()
                else: # ROUND == 1
                    ascent_score = np.ones(N) / 2        # uniformly random suggestion
                    flip_suggest = np.random.randint(6)
                print(f">>> The ALGORITHM says {flip_suggest+1}-th is the best position to flip, because...")
                for i in range(6):
                    print(f">>> the value of flipping {i+1}-th position is {ascent_score[i]:.4f};")
            elif back_to_best:
                flip_suggest = PATH[-left_chance]
                print(f">>> The ALGORITHM says {flip_suggest+1}-th is the best position to flip, because...")
                print(f">>> You are ON A WAY to go back to the so-far-best (reachable) state !:", objective_x)
            print()
            

            # Flip-index Choice (by Player)
            flip_idx = player_decision(N, inputs, best, next_x)
            
            # Compute next state / fitness / contribution improvement
            next_y_temp, ctrbs_diff = nkmodel.fitness_and_contrib_diff(next_x, flip_idx)
            next_x = flip(next_x, flip_idx)
            fit_diff = next_y_temp - float(next_y)
            next_y = np.array([next_y_temp])

            # inputs update
            add_data(inputs, next_x, next_y)
            

            if back_to_best and (ROUND >= chances - N):
                # If the player did different action from the suggestion, initialize PATH
                if flip_suggest != flip_idx:
                    PATH = None 

                # Possibly, update PATH (if it exists)
                if (PATH is not None) and (left_chance-1>0) and ((left_chance-1) % 2 == 0) and (next_y > objective_y):  # new reachable best --> update PATH
                    PATH[-(left_chance-1):] = wander(next_x, left_chance-1)
            
                # At some point, we should save a PATH to reachable best so far.
                if PATH is None: 
                    PATH, objective_y, objective_x = monitor_reachable_best(inputs['x_vals'], inputs['y_vals'], ROUND, chances, show_objective_x=True)
                    print("*#*#* Notice: From now, your on the way to go back to the state:", objective_x)
                    print("*#*#* Of course, you may take different way from the algorithm's suggestion! ")
                    time.sleep(0.5)
            
            # surrogate model train
            LR.fit(inputs['x_vals'], inputs['y_vals'])

            # best state update
            if next_y > best['y']:
                best["y"] = next_y
                best["x"] = next_x
                best["round"] = ROUND
            print("\n"+"="*100+"\n")
        
        time.sleep(2)
        print("THE END: All the chances Ran out!")
        print("Finally, your best attempt is:")
        _show_best(best)
        print()

        print("Also, your FINAL attempt is:")
        print("*** Displaying Your Final Score ***")
        print("final x :", next_x)
        print("final y :", next_y)
        print()

        # Compute reachable best
        reachable_best = None
        for ind in range(len(optlist)):
            opt, optstates = optlist[ind]["fitness"], optlist[ind]["states"]
            for st in optstates:
                if is_reachable(init_x, np.array(st), chances):
                    reachable_best = opt
                    break
            if reachable_best is not None:
                break

        time.sleep(2)
        # Assesment of result
        if abs(float(next_y) - optimum) <= 1e-5:
            print("You have reached the global optimum !!! You made it !!!")
            start = False
        else:
            print("You did not reached the global optimum.")
            if abs(float(next_y) - reachable_best) <= 1e-5:
                print("However, you did your best!! (Your final answer is 'reachable-optimum')")
                start = False
            else:
                print("Well, everyone has their bad days.")
        print()
        
        # Ask whether to restart
        start = start and can_restart
        time.sleep(3)
        if start:
            print("Do you want to try again? (with the same landscape)")
            if _yes_or_no():
                print("Caution: your best score will be removed from the memory\n")
            else:
                start = False
        
        # Scoreboard
        if not start:
            print("Do you want to see the score board? (Caution: It will print 64-line results.")
            if _yes_or_no():
                for i in range(len(optlist)):
                    opt, optstates = optlist[i]["fitness"], optlist[i]["states"]
                    print(f"{i+1}-th optimum: {opt:.5f} {optstates}")


def GAME_V2(args, chances=None, show_interdependence=False, can_restart=False, surrogate_model_name='PolyReg', back_to_best=False, ascent_trial=64):
    from Models.NK_Landscape import binarr2int, int2binarr, NK_landscape
    print("Preparing the game.....")

    N = args.N
    if chances is None:
        chances = args.n_eval
    elif chances != args.n_eval:
        print("args.n_eval changed:", chances)
        args.n_eval = chances
    all_states = states(N)
    
    # Construct NKmodel
    inter_mat, ctrb_map, landscape = NK_landscape(N=N, K=args.K)
    landscape_sort_ind = np.argsort(-landscape[:,-1]) # sort in ascending order
    optimum = landscape[landscape_sort_ind[0]][-1]  # maximum fitness value
    min_fit_ind = landscape_sort_ind[-1]              # index of minimum-fitness-value state
    if show_interdependence:
        print(inter_mat) # Boolean interdependence matrix
    
    # initial state
    init_x = int2binarr(min_fit_ind, width=N)  # start from teh bottom
    init_payoff = landscape[min_fit_ind]
    init_ctrb, init_y = init_payoff[:-1], np.array([init_payoff[-1]])

    # Compute reachable optimum
    reachable_optimum = None
    for idx in landscape_sort_ind:
        if is_reachable(init_x, int2binarr(idx, width=N), chances):
            reachable_optimum = landscape[idx][-1]
            break
    
    play = True
    while play:
        inputs = {'x_vals': init_x.reshape(1,-1), 'y_vals': init_y}
        best = {'x': init_x.copy(), 'y': float(init_y), 'round': 1}  # store so far best result
        curr_ctrb = init_ctrb.copy()

        # Surrogate model
        LR = surrogate_model_dict[surrogate_model_name]
        
        fit_diff = 0                # fitness  difference (previous --> current)       
        ctrb_diff = np.zeros(N)    # contrib. difference (previous --> current)

        next_x = init_x.copy()
        next_y = init_y
        PATH = None
        print()
        for ROUND in range(1, chances+1):
            print(f"*** Round {ROUND}/{chances} ***")
            print( "Previous results:")
            print( "Current state:", next_x, '<== IMPROVED' if ROUND>1 and np.all(best['x'] == next_x) else '')
            print(f"Current fitness: {float(next_y):.4f} ({float(fit_diff):.4f} from before)")
            print( "Current improvement of contributions:", ctrb_diff)
            print()

            # Flip-index suggestion
            left_chance = chances - ROUND + 1
            if PATH is None:
                if ROUND > 1:
                    stat_model = lambda x: LR.predict(x.reshape(1,N))[0]
                    acqs_ = LR.predict(all_states)
                    ascent_scores = neighbor_suggestion(args, stat_model, inputs, left_chance, trials=ascent_trial, tqdm_on=True, calculated_acqs=acqs_, return_scores=True)
                    best_score = ascent_scores.max()
                    idx_ = [i for i in range(N) if ascent_scores[i] == best_score]
                    flip_suggest = np.random.choice(idx_)
                else: # ROUND == 1
                    ascent_scores = np.ones(N) / 2        # uniformly random suggestion
                    flip_suggest = np.random.randint(6)
                print(f">>> The ALGORITHM says {flip_suggest+1}-th is the best position to flip, because...")
                for i in range(6):
                    print(f">>> the value of flipping {i+1}-th position is {ascent_scores[i]:.4f};")
            elif back_to_best:
                flip_suggest = PATH[-left_chance]
                print(f">>> The ALGORITHM says {flip_suggest+1}-th is the best position to flip, because...")
                print(f">>> You are ON A WAY to go back to the so-far-best (reachable) state !:", objective_x)
            print()

            # Flip-index Choice (by Player)
            flip_idx = player_decision(N, inputs, best, next_x)
            
            # Compute next state / fitness / contribution improvement
            next_x = flip(next_x, flip_idx)
            next_payoff = landscape[binarr2int(next_x)]
            curr_ctrb_temp, next_y_temp = next_payoff[:-1], np.array([next_payoff[-1]])
            fit_diff = next_y_temp - next_y
            ctrb_diff = curr_ctrb_temp - curr_ctrb
            curr_ctrb, next_y = curr_ctrb_temp, next_y_temp

            # inputs update
            add_data(inputs, next_x, next_y)

            # if you don't use Back-to-reachable-Best heuristic,
            # this if condition is not necessary
            if back_to_best and (ROUND >= chances - N):
                # If the player did different action from the suggestion, initialize PATH
                if flip_suggest != flip_idx:
                    PATH = None 

                # Possibly, update PATH (if it exists)
                if (PATH is not None) and (left_chance-1>0) and ((left_chance-1) % 2 == 0) and (next_y > objective_y):  # new reachable best --> update PATH
                    PATH[-(left_chance-1):] = wander(next_x, left_chance-1)
            
                # At some point, we should save a PATH to reachable best so far.
                if PATH is None: 
                    PATH, objective_y, objective_x = monitor_reachable_best(inputs['x_vals'], inputs['y_vals'], ROUND, chances, show_objective_x=True)
                    print("*#*#* Notice: From now, your on the way to go back to the state:", objective_x)
                    print("*#*#* Of course, you may take different way from the algorithm's suggestion! ")
                    time.sleep(0.5)
            
            # surrogate model train
            LR.fit(inputs['x_vals'], inputs['y_vals'])

            # best state update
            if next_y > best['y']:
                best["y"] = next_y
                best["x"] = next_x
                best["round"] = ROUND
            print("\n"+"="*100+"\n")
        
        time.sleep(2)
        print("THE END: All the chances Ran out!")
        print("Finally, your best attempt is:")
        _show_best(best)
        print()

        print("Also, your FINAL attempt is:")
        print("*** Displaying Your Final Score ***")
        print("final x :", next_x)
        print("final y :", next_y)
        print()

        time.sleep(2)
        # Assesment of result
        if abs(float(next_y) - optimum) <= 1e-5:
            print("You have reached the global optimum !!! You made it !!!")
            play = False
        else:
            print("You did not reached the global optimum.")
            if abs(float(next_y) - reachable_optimum) <= 1e-5:
                print("However, you did your best!! (Your final answer is 'reachable-optimum')")
                play = False
            else:
                print("Well, everyone has their bad days.")
        print()
        
        # Ask whether to restart
        play = play and can_restart
        time.sleep(3)
        if play:
            print("Do you want to try again? (with the same landscape)")
            if _yes_or_no():
                print("Caution: your best score will be removed from the memory\n")
            else:
                play = False
        
        # Scoreboard
        if not play:
            print("Do you want to see the score board? (Caution: It will print 64-line results.")
            if _yes_or_no():
                for i, idx in enumerate(landscape_sort_ind):
                    opt = landscape[idx][-1]
                    optstate = int2binarr(idx, width=N)
                    print(f"{i+1}-th optimum: {opt:.5f} {optstate}")

if __name__ == "__main__":
    args = EasyDict()

    args.random_seed = 2021
    args.n_eval = 18
    args.n_init = 2
    args.N = 6
    args.K = 1
    args.A = 2
    args.terminal_size = 50

    np.set_printoptions(precision=3)

    # Modify!
    GAME_V2(args, chances=18, show_interdependence=False, surrogate_model_name='PolyReg', back_to_best=False, ascent_trial=64)

    # surrogate_model_name은 'BOCS', 'PolyReg', 'Lasso' 중에서 고를 수 있습니다. 속도와 정확성을 위해 PolyReg 또는 Lasso를 추천합니다. (BOCS가 4배 정도 느립니다.)
    # back_to_best는 True, False 중에서 고를 수 있으며, True 이면 Back-to-Reachable-Best Heuristic을 마지막 N=6번에 적용합니다. False를 추천합니다.
    # ascent_trial은 아무런 자연수를 넣어도 됩니다. 작을수록 빠르지만 부정확하며, 클수록 느리지만 정확합니다.