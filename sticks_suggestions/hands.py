import cv2
import mediapipe as mp

class Hand:
    '''
    A class for mediapipe hand landmarking.
    
    ...
    
    Attributes
    ----------
    model_path : str
        the path to the hand landmarker model
    running_mode : mp.tasks.vision.RunningMode
        the mediapipe running mode, {IMAGE, VIDEO, LIVESTREAM}
    num_hands : int
        the maximum number of hands to detect
    min_hand_detection_confidence : float
        The minimum confidence score for the hand detection to be considered successful in palm detection model.
    min_hand_presence_confidence : float
        The minimum confidence score for the hand presence score in the hand landmark detection model. In Video mode and Live stream mode, if the hand presence confidence score from the hand landmark model is below this threshold, Hand Landmarker triggers the palm detection model.
    min_tracking_confidence : float
        The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame.
    result_callback : function
        Sets the result listener to receive the detection results asynchronously when the hand landmarker is in live stream mode.
    
    Methods
    -------
    find_hands(img):
        Given an image, returns a dictionary containing the current game position.
    '''
    
    def __init__(self, model_path="models/hand_landmarker.task",
                 running_mode="LIVE_STREAM",
                 num_hands=4,
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 result_callback=None):
        '''
        Constructs the attributes for the hand object
        
        Parameters
        ----------
            model_path : str, optional
            the path to the hand landmarker model (Default is "models/hand_landmarker.task")
            running_mode : str, optional
                the mediapipe running mode, {"IMAGE", "VIDEO", "LIVESTREAM"}, (Default is "LIVESTREAM")
            num_hands : int, optional 
                the maximum number of hands to detect (Default is 4)
            min_hand_detection_confidence : float, optional
                The minimum confidence score for the hand detection to be considered successful in palm detection model. (Default is 0.5)
            min_hand_presence_confidence : float, optional
                The minimum confidence score for the hand presence score in the hand landmark detection model. In Video mode and Live stream mode, if the hand presence confidence score from the hand landmark model is below this threshold, Hand Landmarker triggers the palm detection model. (Default is 0.5)
            min_tracking_confidence : float, optional
                The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. (Default is 0.5)
            result_callback : function, optional
                Sets the result listener to receive the detection results asynchronously when the hand landmarker is in live stream mode. (Default is None)
        '''
        
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
        Given an image, returns a dictionary containing the current game position.
        
            Parameters:
                img (numpy_frame): image frame from OpenCV's video input
                
            Returns:
                current_position (dict): dictionary containing the current game position. None if no hands detected.
                Example: {"player_1": {"left": 1, "right": 1}, "player_2": {"left": 1, "right": 1}}
        '''
        pass