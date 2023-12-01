import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(f"{dir_path}/../"))

import time
from sticks_suggestions.algorithm import Algorithm

# testing the algorithm for a single move

DEPTH = 20
SAVE_TT = False
TT_PATH = "tests/tt.pickle"
PLAYER = 'p1'
ASYNC = True
ASYNC_TIME = 10

algo = Algorithm(TT_PATH,DEPTH)


algo.sg.update_position({"p1":{"l":3, "r":3}, "p2":{"l":1, "r":0}})
algo.sg.switch_turn(PLAYER)

algo.get_best_move(async_=ASYNC, player_turn=PLAYER)

if ASYNC_TIME:
    time.sleep(ASYNC_TIME)

if SAVE_TT:
    algo.save_tt()

print(algo.suggested_move)