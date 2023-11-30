import cv2
import sticks_suggestions.algorithm as algorithm
import sticks_suggestions.hands as hands

VIDEO_INPUT_INDEX = 0
RESOLUTION = (720,720)
MAX_DEPTH = 20
SAVE_TT_TO_FILE = False

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
    algo = algorithm.Algorithm(tt_path="tt.pickle",max_depth=MAX_DEPTH)
    
    # camera setup
    cap = cv2.VideoCapture(VIDEO_INPUT_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    
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
        
        # if game position changed and not already calculating a move, find next best move
        if not (algo.negamax_thread and algo.negamax_thread.is_alive()) and not game_position == hand.get_current_gp():
            game_position = hand.get_current_gp()
            algo.get_best_move(game_position,"p1")
        
        # if the suggested move has changed, show new suggestion
        if algo.suggested_move:
            print(f"{algo.gp_for_suggested}: {algo.suggested_move}")
            algo.suggested_move = []
        
        cv2.imshow('image', img)
        
        if cv2.waitKey(1) == ord('q'):
            print("'q' pressed, exiting")
            break 
    
    if SAVE_TT_TO_FILE:  
        algo.save_tt()
    
    # release all assets
    cap.release()
    cv2.destroyAllWindows()
    hand.close_landmarker()