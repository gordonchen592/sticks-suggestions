import cv2
import mediapipe as mp
from sticks_suggestions.hands import Hand

hand = Hand()

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    
    img = hand.find_hands(img)
    
    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
hand.close_landmarker()