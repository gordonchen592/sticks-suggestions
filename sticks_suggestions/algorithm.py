import pickle
from pathlib import Path
import threading
from copy import deepcopy

class Sim_Game:
    def __init__(self):
        self.hand_dict = {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
        self.player_turn = "p1"
        self.player_turn_next = "p2"

    def switch_turn(self, player_turn=None):
        if player_turn:
            self.player_turn = player_turn
            self.player_turn_next = "p2" if player_turn=="p1" else "p1"
        else:
            self.player_turn, self.player_turn_next = self.player_turn_next, self.player_turn
    
    def update_position(self, hand_dict_, player_turn=None):
        self.hand_dict = deepcopy(hand_dict_)
        if player_turn:
            self.switch_turn(player_turn)
    
    def move(self,used_hand=None, target_hand=None, rh_val=None, type="tap"):
        '''
        Calculates the intended move and performs it. Then, if move is valid, switches the player's turn.

        Parameters
        ----------
            used_hand: char, optional
                The hand used by the player. {"l", "r", None} (Default is None in "split" mode and "r" in "tap" mode)
            target_hand: char, optional
                The hand targeted by the player to be hit. {"l", "r", None} (Default is None in "split" mode and "r" in "tap" mode)
            rh_val: int, optional
                The intended count of sticks on the player's right hand after splitting when in "split" mode. An integer 0 through 4, or None (Default is None in "tap"mode and "1" in split mode)
            type: str, optional
                The type of move. {"tap","split"} (Default is "tap")
        Returns
        -------
            boolean
                True if a valid move is made
                False if an invalid move is made
        '''
        # Tapping and Splitting
        if type == "tap":
            if (0 < self.hand_dict[self.player_turn][used_hand] < 5) and (0 < self.hand_dict[self.player_turn_next][target_hand] < 5):
                self.hand_dict[self.player_turn_next][target_hand] += self.hand_dict[self.player_turn][used_hand]
                self.hand_dict[self.player_turn_next][target_hand] %= 5 # max sticks per hand is 5; implements roll-over rule
                self.player_turn, self.player_turn_next = self.player_turn_next, self.player_turn
                return True
            else:
                return False
        elif type == "split":
            if sum(self.hand_dict[self.player_turn].values()) > 1 and self.hand_dict[self.player_turn]["l"] != rh_val:
                sticks_moved = self.hand_dict[self.player_turn]["r"] - rh_val
                if (sticks_moved + self.hand_dict[self.player_turn]["l"])%5 == rh_val == 0 or sticks_moved + self.hand_dict[self.player_turn]["l"] < 0:
                    return False
                self.hand_dict[self.player_turn]["r"] = rh_val
                self.hand_dict[self.player_turn]["l"] += sticks_moved
                self.hand_dict[self.player_turn]["l"] %= 5
                self.player_turn, self.player_turn_next = self.player_turn_next, self.player_turn
                return True
            else:
                return False
    def is_p1_win(self):
        return self.hand_dict['p2']['l'] == self.hand_dict['p2']['r'] == 0
    def is_p2_win(self):
        return self.hand_dict['p1']['l'] == self.hand_dict['p1']['r'] == 0
    def is_end(self):
        return self.is_p1_win() or self.is_p2_win()
    def get_gp_as_str(self):
        return str(self.hand_dict['p1']['l']) + str(self.hand_dict['p1']['r']) + str(self.hand_dict['p2']['l']) + str(self.hand_dict['p2']['r'])
    def get_possible_moves(self):
        possible_list = []
        
        # tapping
        for used_hand in ['r','l']:
            for target_hand in ['r','l']:
                if (0 < self.hand_dict[self.player_turn][used_hand] < 5) and (0 < self.hand_dict[self.player_turn_next][target_hand] < 5):
                    possible_list.append(f"tap {used_hand} {target_hand}")
        
        # splitting
        for rh_val in range(0,5):
            if sum(self.hand_dict[self.player_turn].values()) > 1 and self.hand_dict[self.player_turn]["l"] != rh_val:
                sticks_moved = self.hand_dict[self.player_turn]["r"] - rh_val
                if (sticks_moved + self.hand_dict[self.player_turn]["l"])%5 == rh_val == 0 or sticks_moved + self.hand_dict[self.player_turn]["l"] < 0:
                    continue
                possible_list.append(f"split {rh_val}")
        return possible_list

class Algorithm:
    LOWERBOUND,EXACT,UPPERBOUND = -1,0,1
    inf = float('infinity')
    
    def __init__(self, tt_path, max_depth):
        self.tt_path = tt_path
        self.max_depth = max_depth
        self.suggested_move = []
        self.sg = Sim_Game()
        
        self.result_lock = threading.Lock()
        self.negamax_thread = None
        
        # make transposition table file and load the table
        file_path = Path(self.tt_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.tt_path, 'rb') as f:
                self.tt = pickle.load(f)
        except:
                self.tt = {}
    
    def save_tt(self):
        with open(self.tt_path, 'wb') as f:
            self.tt = pickle.dump(self.tt, f)
    
    def get_best_move(self, hand_dict=None, player_turn=None, async_=True):
        if hand_dict:
            self.sg.update_position(hand_dict)
        if player_turn:
            self.sg.switch_turn(player_turn)
        if async_:
            if not (self.negamax_thread and self.negamax_thread.is_alive()):
                self.negamax_thread = threading.Thread(target=self.negamax,args=(self.sg,self.max_depth,))
                self.negamax_thread.start()
        else:
            self.negamax(self.sg,self.max_depth)
    
    def negamax(self, s_game:Sim_Game, depth, alpha=-inf, beta=inf, color=1):
        '''
        Converted from the [Wikipedia page for Negamax with alpha beta pruning and transposition tables](https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning_and_transposition_tables)
        '''
        alpha_orig = alpha
        gp = s_game.get_gp_as_str()
        
        # transposition table lookup
        tt_entry = self.tt.get(gp)
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == Algorithm.EXACT:
                if depth == self.max_depth:
                    self.suggested_move = tt_entry['move']
                return tt_entry['value']
            elif tt_entry['flag'] == Algorithm.LOWERBOUND:
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == Algorithm.UPPERBOUND:
                beta = min(beta, tt_entry['value'])
            
            if alpha >= beta:
                if depth == self.max_depth:
                    self.suggested_move = tt_entry['move']
                return tt_entry['value']
        
        # calculate node value
        if depth == 0 or s_game.is_end():
            if s_game.is_p1_win():
                value = 1000
            elif s_game.is_p2_win():
                value = -1000
            else:
                value = 0
            return int(color*value*(1+depth*0.001))
        
        child_nodes = s_game.get_possible_moves()
        best_value = -1*Algorithm.inf
        best_move = []
        s_game_copy:Sim_Game = deepcopy(s_game)
        hd = deepcopy(s_game.hand_dict)
        for move in child_nodes:
            move_ = move.split(" ")
            if move_[0] == 'tap':
                valid = s_game_copy.move(type='tap',used_hand=move_[1],target_hand=move_[2])
            else:
                valid = s_game_copy.move(type='split',rh_val=int(move_[1]))
            if not valid:
                s_game_copy.update_position(hd)
                s_game_copy.switch_turn()
                continue
            
            value = -self.negamax(s_game_copy, depth -1, -beta, -alpha, -color)
            
            if best_value < value:
                best_value = value
                best_move = move_

            if alpha < value:
                alpha = value
                if depth == self.max_depth:
                    self.suggested_move = move_
                if alpha >= beta:
                    break
            
            # reset move for next possible move in for loop
            s_game_copy.update_position(hd)
            s_game_copy.switch_turn()
        
        # store in transposition table
        tt_entry = {}
        tt_entry['value'] = best_value
        if best_value <= alpha_orig:
            tt_entry['flag'] = Algorithm.UPPERBOUND
        elif best_value >= beta:
            tt_entry['flag'] = Algorithm.LOWERBOUND
        else:
            tt_entry['flag'] = Algorithm.EXACT
        tt_entry['depth'] = depth
        tt_entry['move'] = best_move
        self.tt[gp] = tt_entry
        
        return best_value