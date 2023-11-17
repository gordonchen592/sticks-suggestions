import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(f"{dir_path}/../"))

from sticks_suggestions.algorithm import Sim_Game

sg = Sim_Game()

print("Controls:\n\tTapping an Opponent: 'tap [use hand, r/l] [target hand, r/l]'\n\tSplitting: 'split [intended right hand value, int]'\n\tQuitting: 'q'")

move = [""]

while not move[0] == 'q':
    
    while not move[0] == 'q':
        print(f"P1: L{sg.hand_dict['p1']['l']} R{sg.hand_dict['p1']['r']}\tP2: L{sg.hand_dict['p2']['l']} R{sg.hand_dict['p2']['r']}")
        move = input("P1 Intended Move (ex. 'tap l r' or 'split 2'): ").split(" ")
        try:
            if move[0] == 'q':
                break
            elif move[0] == 'tap':
                valid = sg.move(type='tap',used_hand=move[1],target_hand=move[2])
            else:
                valid = sg.move(type='split',rh_val=int(move[1]))
            if valid:
                break
            else:
                print("Invalid move")
        except:
            print("invalid input or other error")
    sg.switch_turn()
    
    while not move[0] == 'q':
        print(f"P1: L{sg.hand_dict['p1']['l']} R{sg.hand_dict['p1']['r']}\tP2: L{sg.hand_dict['p2']['l']} R{sg.hand_dict['p2']['r']}")
        move = input("P2 Intended Move (ex. 'tap l r' or 'split 2'): ").split(" ")
        try:
            if move[0] == 'q':
                break
            elif move[0] == 'tap':
                valid = sg.move(type='tap',used_hand=move[1],target_hand=move[2])
            else:
                valid = sg.move(type='split',rh_val=int(move[1]))
            if valid:
                break
            else:
                print("Invalid move")
        except:
            print("invalid input or other error")
    sg.switch_turn()