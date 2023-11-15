import numpy as py

class Sim_Game:
    def __init__(self):
        self.hand_dict = {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
        self.player_turn = "p1"
        self.player_turn_next = "p2"

    def switch_turn(self, player_turn=None):
        if player_turn:
            self.player_turn = "p1"
            self.player_turn_next = "p2" if player_turn=="p1" else "p1"
    
    def update_position(self, hand_dict_, player_turn=None):
        self.hand_dict = hand_dict_
        if player_turn:
            self.switch_turn(player_turn)
    
    def move(self,used_hand=None, target_hand=None, rh_val=None, type="tap"):
        '''
        Calculates the intended move and performs it.

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
                if sticks_moved + self.hand_dict[self.player_turn]["l"] == 0 and rh_val == 0:
                    return False
                self.hand_dict[self.player_turn]["r"] = rh_val
                self.hand_dict[self.player_turn]["l"] += sticks_moved
                return True
            else:
                return False


