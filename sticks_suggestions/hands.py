import cv2
import mediapipe as mp

class Hand:
    
    def __init__(self, model_path="models/hand_landmarker.task",
                 running_mode="LIVE_STREAM",
                 num_hands=4,
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 result_callback=None):
        # init instance variables
        self.model_path = model_path
        match(running_mode):
            case "IMAGE":
                self.running_mode = mp.tasks.vision.RunningMode.IMAGE
            case "VIDEO":
                self.running_mode = mp.tasks.vision.RunningMode.VIDEO
            case _: # LIVE_STREAM
                self.running_mode = mp.tasks.vision.RunningMode.LIVESTREAM
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.result_callback = result_callback
        
 
        # Set up the Hands Landmarker
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            result_callback=self.result_callback)
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    
    def find_hands(self,img):
        '''
        Return a dictionary containing the current game position:
        Example: {"player_1": {"left": 1, "right": 1}, "player_2": {"left": 1, "right": 1}}
        '''
        pass