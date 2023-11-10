import numpy as py
import hands

class Sim_Game:
    def __init__(self):
        self.hand_dict = {"p1":{"l":1, "r":1}, "p2":{"l":1, "r":1}}
        self.person_turn = "p1"
        self.person_next_turn = "p2"
        pass

    def move(self,used_hand=None, target_hand=None, Right_hand=None,type="tap"):
        '''
        Calculates the intended move.

        Optional Parameters:
        used_hand: "l" or "r" 
        target_hand: "l" or "r" 
        Right_hand: 0 to 4 (Step = 1)
        '''
        if type == "tap":
            if (0 < self.hand_dict[self.person_turn][used_hand] < 5) and (0 < self.hand_dict[self.person_next_turn][target_hand] < 5):
                self.hand_dict[self.person_next_turn][target_hand] += self.hand_dict[self.person_turn][used_hand]
                self.person_turn, self.person_next_turn = self.person_next_turn, self.person_turn
                return True
            else:
                return False