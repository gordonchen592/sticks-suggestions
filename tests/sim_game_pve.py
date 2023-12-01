import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(f"{dir_path}/../"))

from sticks_suggestions.algorithm import Sim_Game
from sticks_suggestions.algorithm import Algorithm

# Player vs Computer
# Used for testing the rules of the game.   

DEPTH = 20
SAVE_TT = False
TT_PATH = "tests/tt.pickle"
PLAYER = 'p1'

algo = Algorithm(TT_PATH, DEPTH)

print("Controls:\n\tTapping an Opponent: 'tap [use hand, r/l] [target hand, r/l]'\n\tSplitting: 'split [intended right hand value, int]'\n\tQuitting: 'q'")

move = [""]
player = ""

# assign who goes first
if PLAYER:
    player = PLAYER
else:
    while not player:
        player = input("Which player do you want to be? Player 1 goes first. ('P1' or 'P2'):").lower()
        if not (player.lower() == 'p1' or player.lower() == 'p2'):
            print("Invalid player. Possible players are 'P1' or 'P2'.")
            player = ""

cpu = 'p1' if player != 'p1' else 'p2'

while not move[0] == 'q':
    # print game state
    print(f"{player}: L{algo.sg.hand_dict[player]['l']} R{algo.sg.hand_dict[player]['r']}\tCPU: L{algo.sg.hand_dict[cpu]['l']} R{algo.sg.hand_dict[cpu]['r']}")
    
    # check if game is over
    if algo.sg.is_end():
        if (algo.sg.is_p1_win() and player == 'p1') or (algo.sg.is_p2_win() and player == 'p2'):
            print("You Win!")
            if SAVE_TT:
                algo.save_tt()
            exit()
        else :
            print("You Lose...")
            if SAVE_TT:
                algo.save_tt()
            exit()
    
    # if its the player's turn
    if algo.sg.player_turn == player:
        while not move[0] == 'q':
            move = input("Player Intended Move (ex. 'tap l r' or 'split 2'): ").split(" ")
            try:
                if move[0] == 'q':
                    if SAVE_TT:
                        algo.save_tt()
                    exit()
                elif move[0] == 'tap':
                    valid = algo.sg.move(type='tap',used_hand=move[1],target_hand=move[2])
                else:
                    valid = algo.sg.move(type='split',rh_val=int(move[1]))
                if valid:
                    break
                else:
                    print("Invalid move")
            except:
                print("invalid input or other error")
    else:
        move = algo.get_best_move(async_=False, player_turn=cpu)
        
        print(f"CPU perfroms a {move[0]}, {f'using its {move[1]} to attack your {move[2]}' if move[0]=='tap' else f'splitting its right hand into {move[1]}'}")
        
        if move[0] == 'tap':
            valid = algo.sg.move(type='tap',used_hand=move[1],target_hand=move[2])
        else:
            valid = algo.sg.move(type='split',rh_val=int(move[1]))
if SAVE_TT:
    algo.save_tt()