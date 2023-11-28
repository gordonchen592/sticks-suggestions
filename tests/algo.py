import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(f"{dir_path}/../"))

import time
from sticks_suggestions.algorithm import Algorithm

algo = Algorithm("tests/tt.pickle",20)

algo.sg.update_position({"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}})
algo.sg.switch_turn("p1")

algo.get_best_move(async_=False)

algo.save_tt()

print(algo.suggested_move)