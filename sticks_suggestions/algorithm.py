import pickle
from pathlib import Path
import threading
from copy import deepcopy

class Sim_Game:
    '''
    A class for simulating a game of [chopsticks (AKA sticks or calculator)](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)).
    
    ...
    
    Attributes
    ----------
    hand_dict : dict
        The current game position as a dictionary. Example: {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
    player_turn : str
        The current player whose turn it is.
    player_turn_next : str
        The player whose turn is after the current player's turn.
    
    Methods
    -------
    switch_turn(player_turn):
        Toggles which player performs the next move command and which player is after that move. Can also be used to set the player given player string.
    update_position(hand_dict_, player_turn):
        Sets the game position to the given position dictionary.
    move(used_hand, target_hand, rh_val, type):
        Calculates the intended move and performs it. Then, if move is valid, switches the player's turn.
    is_p1_win():
        Returns whether the current game position is a winning position for player 1.
    is_p2_win():
        Returns whether the current game position is a winning position for player 1.
    is_end():
        Returns whether the current game position is an end position where no valid moves can be made.
    get_gp_as_str():
        Returns the current game position as a string in the xxxx format.
    get_possible_moves():
        Returns a list of the possible moves for the current game position in a string format.
    '''
    
    def __init__(self, hand_dict:dict={"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}, player_turn:str="p1"):
        '''
        Constructs the attributes for the sim_game object
        
        Parameters
        ----------
            hand_dict : dict, optional
                the current game position as a dictionary (Default is {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}})
            player_turn : str, optional
                the current player whose turn it is (Default is "p1")
        '''
        if (hand_dict):
            self.hand_dict = hand_dict
        else:
            self.hand_dict = {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
            
        if (player_turn):
            self.player_turn = player_turn
            self.player_turn_next = "p2" if player_turn=="p1" else "p1"
        else:
            self.player_turn = "p1"
            self.player_turn_next = "p2"

    def switch_turn(self, player_turn:str=None):
        '''
        Toggles which player performs the next move command and which player is after that move. Can also be used to set the player given player string.
        
        Parameters
        ----------
            player_turn : str, optional
                the player to set for the current turn. {"p1","p2", None} (Default is None which toggles whose turn it is)
        '''
        if player_turn:
            self.player_turn = player_turn
            self.player_turn_next = "p2" if player_turn=="p1" else "p1"
        else:
            self.player_turn, self.player_turn_next = self.player_turn_next, self.player_turn
    
    def update_position(self, hand_dict_:dict, player_turn:str=None):
        '''
        Sets the game position to the given position dictionary.
        
        Parameters
        ----------
            hand_dict_ : dict
                the game position dictionary to set as the current position. Example: {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
            player_turn : str, optional
                the player to set for the current turn. {"p1","p2", None} (Default is None which leaves whose turn it is as is)
        '''
        self.hand_dict = deepcopy(hand_dict_)
        if player_turn:
            self.switch_turn(player_turn)
    
    def move(self,used_hand:str=None, target_hand:str=None, rh_val:int=None, type:str="tap") -> bool:
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
            bool
                True if a valid move is made
                False if an invalid move is made
        '''
        # Tapping and Splitting
        if type == "tap":
            # no move specified
            if not (used_hand and target_hand):
                return False
            # check if move is valid
            if (0 < self.hand_dict[self.player_turn][used_hand] < 5) and (0 < self.hand_dict[self.player_turn_next][target_hand] < 5):
                self.hand_dict[self.player_turn_next][target_hand] += self.hand_dict[self.player_turn][used_hand]
                self.hand_dict[self.player_turn_next][target_hand] %= 5 # max sticks per hand is 5; implements roll-over rule
                self.player_turn, self.player_turn_next = self.player_turn_next, self.player_turn
                return True
            else:
                return False
        elif type == "split":
            # no split value specified
            if rh_val == None:
                return False
            # if the resulting count is more than 1 and is not a pass (switching numbers from r to l)
            if sum(self.hand_dict[self.player_turn].values()) > 1 and not self.hand_dict[self.player_turn]["l"] == rh_val:
                sticks_moved = self.hand_dict[self.player_turn]["r"] - rh_val
                # if no change or both hands end up as zero or left hand requires negative value
                if sticks_moved == 0 or ((sticks_moved + self.hand_dict[self.player_turn]["l"])%5 == rh_val == 0) or sticks_moved + self.hand_dict[self.player_turn]["l"] < 0:
                    return False
                self.hand_dict[self.player_turn]["r"] = rh_val
                self.hand_dict[self.player_turn]["l"] += sticks_moved
                self.hand_dict[self.player_turn]["l"] %= 5
                self.player_turn, self.player_turn_next = self.player_turn_next, self.player_turn
                return True
            else:
                return False
    def is_p1_win(self) -> bool:
        return self.hand_dict['p2']['l'] == self.hand_dict['p2']['r'] == 0
    def is_p2_win(self) -> bool:
        return self.hand_dict['p1']['l'] == self.hand_dict['p1']['r'] == 0
    def is_end(self) -> bool:
        return self.is_p1_win() or self.is_p2_win()
    def get_gp_as_str(self) -> str:
        '''
        Returns the current game position as a string in the xxxx format.

        Returns
        -------
            str
                A string representing the current game position in the format xxxx where each x is a digit from 0 to 5. The digits represent the hand value of each hand for each player.
                From left to right, the digits represent: player 1's left hand, player 1's right hand, player 2's left hand, player 2's right hand. In short hand: [P1L][P1R][P2L][P2R].
        '''
        return str(self.hand_dict['p1']['l']) + str(self.hand_dict['p1']['r']) + str(self.hand_dict['p2']['l']) + str(self.hand_dict['p2']['r'])
    def get_possible_moves(self) -> list:
        '''
        Returns a list of the possible moves for the current game position in a string format.

        Returns
        -------
            str
                A list of strings each representing the possible moves for the current game position.
                Moves come in two varieties: tapping or splitting.
                
                For tapping, the format is "tap {used_hand} {target_hand}".
                    Example: "tap l r" means the current player taps the opponent's right hand using their left hand.
                
                For splitting, the format is "split {rh_val}".
                    Example: "split 2" means the current player attempts to split their left and right hands such that the right hand ends with a value of 2.
        '''
        possible_list = []
        
        # tapping
        for used_hand in ['r','l']:
            for target_hand in ['r','l']:
                if (0 < self.hand_dict[self.player_turn][used_hand] < 5) and (0 < self.hand_dict[self.player_turn_next][target_hand] < 5):
                    possible_list.append(f"tap {used_hand} {target_hand}")
        
        # splitting
        for rh_val in range(0,5):
            # if the resulting count is more than 1 and is not a pass (switching numbers from r to l)
            if sum(self.hand_dict[self.player_turn].values()) > 1 and self.hand_dict[self.player_turn]["l"] != rh_val:
                sticks_moved = self.hand_dict[self.player_turn]["r"] - rh_val
                print(self.hand_dict[self.player_turn]["r"])
                # if no change or both hands end up as zero or left hand requires negative value
                if sticks_moved == 0 or (sticks_moved + self.hand_dict[self.player_turn]["l"])%5 == rh_val == 0 or sticks_moved + self.hand_dict[self.player_turn]["l"] < 0:
                    continue
                possible_list.append(f"split {rh_val}")
        return possible_list

class Algorithm:
    '''
    A class for finding the next best move for a game of chopsticks using the negamax algorithm with alpha/beta pruning and transposition tables.
    
    ...
    
    Attributes
    ----------
    tt_path : str
        The path to the transposition table file.
    max_depth : int
        The maximum future moves for which the negamax algorithm should look ahead
    suggested_move : list
        A list containing the suggested next best move for player 1 based on the current game position
    gp_for_suggested : str
        A string representing the game position associated with the suggested_move in the xxxx format
    sg : Sim_Game
        The Sim_Game object used by the algorithm
    negamax_thread : Thread
        The thread used for the async get_best_move() function
    
    Methods
    -------
    save_tt():
        Saves the transposition table to the file system
    get_best_move(hand_dict, player_turn, async_):
        Commands the algorithm to find the next best move for player 1 based on the game position. Can be called asyncronously.
    negamax(s_game, depth, alpha, beta, color):
        The negamax algorithm with alpha-beta pruning and transposition tables. (Recursive)
    '''
    LOWERBOUND,EXACT,UPPERBOUND = -1,0,1
    inf = float('infinity')
    
    def __init__(self, tt_path:str, max_depth:int):
        self.tt_path = tt_path
        self.max_depth = max_depth
        self.suggested_move = []
        self.gp_for_suggested = ""
        self.sg = Sim_Game()
        
        self.negamax_thread = None
        
        # make transposition table file and load the table
        file_path = Path(self.tt_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.tt = {}
        try:
            with open(self.tt_path, 'rb') as f:
                self.tt = pickle.load(f)
        except:
            pass
    
    def save_tt(self):
        '''
        Saves the transposition table to the file system
        '''
        with open(self.tt_path, 'wb') as f:
            pickle.dump(self.tt, f)
    
    def get_best_move(self, hand_dict:dict=None, player_turn:str=None, async_:bool=True) -> bool|list:
        '''
        Commands the algorithm to find the next best move for player 1 based on the game position. Can be called asynchronously.
        
            Parameters
                hand_dict_ : dict, optional
                    the game position dictionary that the next move should be searched on (Default is None which uses the current game position in the built in Sim_Game). 
                    Example: {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
                player_turn : str, optional
                    the player whose turn it should be when searching the next move {"p1","p2",None} (Default is None which uses player 1).
                async_ : bool, optional
                    whether to search asynchronously or not (Default is True)
            
            Returns
                bool
                    False if asyncronous but an existing calculation has not completed yet
                    True if asyncronous and the existing calculation was started successfully
                list
                    A list containing the suggested move commands in the format [move, **move_parameters] where the move and move_parameters fit the Sim_Game class move() function. Only when running in non-async mode
        '''
        if async_:
            if not (self.negamax_thread and self.negamax_thread.is_alive()):
                if hand_dict:
                    self.sg.update_position(hand_dict)
                if player_turn:
                    self.sg.switch_turn(player_turn)
                self.negamax_thread = threading.Thread(target=self.negamax, args=(deepcopy(self.sg),self.max_depth),kwargs={'color': 1 if player_turn=='p1' else -1})
                self.negamax_thread.start()
                return True
            else:
                return False
        else:
            if hand_dict:
                self.sg.update_position(hand_dict)
            if player_turn:
                self.sg.switch_turn(player_turn)
            self.negamax(self.sg,self.max_depth, color=1 if player_turn=='p1' else -1)
            return self.suggested_move
    
    def negamax(self, s_game:Sim_Game, depth:int, alpha:float=-inf, beta:float=inf, color:float=1) -> int|float:
        '''
        Converted from the [Wikipedia page for Negamax with alpha-beta pruning and transposition tables](https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning_and_transposition_tables)
        '''
        print(color)
        alpha_orig = alpha
        gp = s_game.get_gp_as_str()
        pt = str(s_game.player_turn)
        
        # transposition table lookup
        tt_entry = self.tt.get(pt+gp)
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == Algorithm.EXACT:
                if depth == self.max_depth:
                    self.suggested_move = tt_entry['move']
                    self.gp_for_suggested = self.sg.get_gp_as_str()
                return tt_entry['value']
            elif tt_entry['flag'] == Algorithm.LOWERBOUND:
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == Algorithm.UPPERBOUND:
                beta = min(beta, tt_entry['value'])
            
            if alpha >= beta:
                if depth == self.max_depth:
                    self.suggested_move = tt_entry['move']
                    self.gp_for_suggested = self.sg.get_gp_as_str()
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
            print(s_game_copy.get_gp_as_str(),s_game_copy.player_turn,color,":",str(move_))
            if move_[0] == 'tap':
                valid = s_game_copy.move(type='tap',used_hand=move_[1],target_hand=move_[2])
            else:
                valid = s_game_copy.move(type='split',rh_val=int(move_[1]))
            if not valid:
                print("not valid")
                s_game_copy.update_position(hd)
                s_game_copy.switch_turn()
                continue
            
            value = -self.negamax(s_game_copy, depth -1, -beta, -alpha, -color)
            
            if best_value < value:
                best_value = value
                best_move = move_.copy()

            if alpha < value:
                alpha = value
                if depth == self.max_depth:
                    self.suggested_move = move_.copy()
                    self.gp_for_suggested = self.sg.get_gp_as_str()
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
        self.tt[pt+gp] = tt_entry
        
        return best_value