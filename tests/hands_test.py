import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(f"{dir_path}/../"))

import cv2
import mediapipe as mp
from sticks_suggestions.hands import Hand

hand = Hand(running_mode="LIVE_STREAM",
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.1)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    hand.find_hands(img)
    img = hand.draw_on_image(img,draw_landmarks=False,draw_count=True,draw_player=True, draw_handedness=True)
    
    # print(hand.game_position["p1"]["l"], "\t",hand.game_position["p1"]["r"])
    
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        print("'q' pressed, exiting")
        break
cap.release()
cv2.destroyAllWindows()
hand.close_landmarker()