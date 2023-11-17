import sticks_suggestions.algorithm as algorithm
import sticks_suggestions.hands as hands
import sticks_suggestions.suggest as suggest

import cv2

if __name__ == '__main__':
    # hand detection setup
    hand = hands.Hand(running_mode="LIVE_STREAM",
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.1,
            min_game_position_confidence=0.5,
            max_buffer=5)
    game_position = None
    
    # algorithm setup
    algo = algorithm.Algorithm()
    suggested_move = None
    
    # camera setup
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # wait for camera response
    success, img = cap.read()
    while not success:
        success, img = cap.read()
    
    # video stream
    while True:
        success, img = cap.read()
        
        # find hands, calculate the current game position, draw position on image
        hand.find_hands(img)
        img = hand.draw_on_image(img, draw_count=True,draw_player=True, draw_handedness=True)
        
        # if game position changed, find next best move
        if not game_position == hand.game_position_buffered:
            game_position = hand.game_position_buffered
            algo.get_best_move(game_position,'p1')
        
        # if the suggested move has changed, show new suggestion
        if not suggested_move == algo.suggested_move:
            suggested_move = algo.suggested_move
            print(suggested_move)
        
        cv2.imshow('image', img)
        
        if cv2.waitKey(1) == ord('q'):
            print("'q' pressed, exiting")
            break
    
    # release all assets
    cap.release()
    cv2.destroyAllWindows()
    hand.close_landmarker()