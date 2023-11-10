import cv2
import mediapipe as mp
import time

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
    game_position : dict
        the current game position, updated asynchronously. None if no hands detected.
        Example: {"player_1": {"left": 1, "right": 1}, "player_2": {"left": 1, "right": 1}}
    Methods
    -------
    create_landmarker():
        Creates a hand landmarker object for use in detecting hand landmarks
    find_hands(img):
        Given an image, returns a dictionary containing the current game position.
    calc_game_position():
        Calculates the current game position given the mediapipe handlandmarker result data.
    close_landmarker():
        Closes the hand landmarker object.
    '''
    
    def __init__(self, model_path="models/hand_landmarker.task",
                 running_mode="LIVE_STREAM",
                 num_hands=4,
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5):
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
        
        self.game_position = {}
        self.createLandmarker()

    def createLandmarker(self):
        '''
        Creates a hand landmarker object for this hand instance based on the options set for this hand instance
        '''
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            result_callback=self.calc_game_position_async)
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    
    def find_hands(self,img, video_timestamp=None):
        '''
        Given an image, video, or live stream frame uses mediapipe to determine the hand results 
        and sends it to calc_game_position to update the current game position.
        
            Parameters:
                img : numpy.ndarray
                    image frame from OpenCV's image/video/stream input
                video_timestamp : int, optional
                    the interger timestamp of the video frame in milliseconds. 
                    Only used for video input (not video stream input) (default is None)
        '''
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

        match(self.running_mode):
            case mp.tasks.vision.RunningMode.LIVESTREAM:
                self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
            case mp.tasks.vision.RunningMode.VIDEO:
                a = self.landmarker.detect_for_video(image = mp_image, timestamp_ms = video_timestamp)
                self.calc_game_position(a)
            case _:
                a = self.landmarker.detect(mp_image)
                self.calc_game_position(a)

    def calc_game_position(self, result):
        '''
        Calculates the current game position given the mediapipe handlandmarker result data.
        Used as the async result_callback function for mediapipe's livestream detection.
        
            Parameters:
                result : HandLandmarkerResult
                    The current hand landmark data for the current position. 
                    The result of detecting hand landmarkers from a frame using mediapipe.
        '''
        pass
    def close_landmarker(self):
        '''
        Closes the hand landmarker object.
        '''
        self.landmarker.close()