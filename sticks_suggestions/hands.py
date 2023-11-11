import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time

class Hand:
    '''
    A class for mediapipe hand landmarking. 
    Based on [Google MediaPipe Solutions: Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python) and [Google MediaPipe Solutions: Hand Landmarker Code Example](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
    
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
    result : mp.tasks.vision.HandLandmarkerResult
        The mediapipe hand landmark result containing information about the hand landmarks
    hands : dict
        A dictionary containing the indices in the mediapipe handlandmaker result corresponding to each hand in game
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
    
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    FINGER_INDICES = [[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP],
                      [mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP],
                      [mp.solutions.hands.HandLandmark.RING_FINGER_TIP,mp.solutions.hands.HandLandmark.RING_FINGER_PIP,mp.solutions.hands.HandLandmark.RING_FINGER_MCP],
                      [mp.solutions.hands.HandLandmark.PINKY_TIP,mp.solutions.hands.HandLandmark.PINKY_PIP,mp.solutions.hands.HandLandmark.PINKY_MCP]]
    THUMB_INDICES = [mp.solutions.hands.HandLandmark.THUMB_TIP,mp.solutions.hands.HandLandmark.THUMB_MCP]
    WRIST_INDEX = mp.solutions.hands.HandLandmark.WRIST
    
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
                self.running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.result = []
        self.hands = {"p1": {"l":-1, "r":-1}, "p2": {"l":-1, "r":-1}}
        self.game_position = {"p1": {"l":0, "r":0},"p2": {"l":0, "r":0}}
        self.createLandmarker()

    def createLandmarker(self):
        '''
        Creates a hand landmarker object for this hand instance based on the options set for this hand instance
        '''
        def calc_game_position_async(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.calc_game_position(result)
        
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            result_callback= calc_game_position_async if self.running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM else None)
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
            case mp.tasks.vision.RunningMode.LIVE_STREAM:
                self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
            case mp.tasks.vision.RunningMode.VIDEO:
                a = self.landmarker.detect_for_video(image = mp_image, timestamp_ms = video_timestamp)
                self.calc_game_position(a)
            case _:
                a = self.landmarker.detect(mp_image)
                self.calc_game_position(a)

    def calc_game_position(self, result: mp.tasks.vision.HandLandmarkerResult):
        '''
        Calculates the current game position given the mediapipe handlandmarker result data.
        
            Parameters:
                result : HandLandmarkerResult
                    The current hand landmark data for the current position. 
                    The result of detecting hand landmarkers from a frame using mediapipe.
        '''
        self.result = result
        
        # if no hands found, return None
        if not result.hand_landmarks:
            return None
        
        self.hands = {"p1": {"l":-1, "r":-1}, "p2": {"l":-1, "r":-1}}
        
        # associate hands to players
        #   temporarily assign first found left and ff right to p1 until fully implemented
        for hand_index in range(len(result.hand_landmarks)):
            if(result.handedness[hand_index][0].category_name == "Left" and self.hands["p1"]["l"] == -1):
                self.hands["p1"]["l"] = hand_index
            elif self.hands["p1"]["r"] == -1:
                self.hands["p1"]["r"] = hand_index
        
        # calculate number on each hand
        #   setup variables
        tip = np.zeros(3)
        mcp = np.zeros(3)
        wrist = np.zeros(3)
        
        m_mcp = np.zeros(3)
        i_mcp = np.zeros(3)
        #   calc number on each hand for each hand
        for player in self.hands:
            for hand in self.hands[player]:
                # if no hand detected, assume dead (0)
                if(self.hands[player][hand] == -1):
                    self.game_position[player][hand] = 0
                    continue
                
                count = 0
                hl = result.hand_landmarks[self.hands[player][hand]]
                # build reference (wrist)
                wrist[:] = hl[Hand.WRIST_INDEX].x, hl[Hand.WRIST_INDEX].y, hl[Hand.WRIST_INDEX].z
                
                # count fingers
                for finger in Hand.FINGER_INDICES:
                    # get finger position
                    tip[:] = hl[finger[0]].x, hl[finger[0]].y, hl[finger[0]].z
                    mcp[:] = hl[finger[1]].x, hl[finger[1]].y, hl[finger[1]].z
                    
                    # if angle between wrist-mcp line and mcp-tip line is greater than 135, finger is up
                    d_wrist_mcp = wrist-mcp
                    d_tip_mcp = tip-mcp
                    angle = np.arccos(np.dot(d_wrist_mcp,d_tip_mcp)/(np.linalg.norm(d_wrist_mcp)*np.linalg.norm(d_tip_mcp)))
                    if (angle > 140/180*np.pi):
                        count += 1
                
                # count thumb
                #   positions
                tip[:] = hl[Hand.THUMB_INDICES[0]].x, hl[Hand.THUMB_INDICES[0]].y, hl[Hand.THUMB_INDICES[0]].z
                # pip[:] = hl[Hand.FINGER_INDICES[0][1]].x, hl[Hand.FINGER_INDICES[0][1]].y
                i_mcp[:] = hl[Hand.FINGER_INDICES[0][2]].x, hl[Hand.FINGER_INDICES[0][2]].y, hl[Hand.FINGER_INDICES[0][2]].z
                m_mcp[:] = hl[Hand.FINGER_INDICES[1][2]].x, hl[Hand.FINGER_INDICES[0][2]].y, hl[Hand.FINGER_INDICES[0][2]].z
                
                #   normal vector of reference plane
                n_ref = np.cross(m_mcp,i_mcp)
                #   normal vector of dividing plane
                n_div = np.cross(n_ref,wrist-i_mcp)
                #   solve for 4th parameter for dividing plane eqn
                d = - np.dot(n_div,i_mcp)
                #   sign of m_mcp
                m_mcp_sign = (d + np.dot(m_mcp,n_div)) > 0
                #   sign of t_tip
                tip_loc = (d + np.dot(tip,n_div))
                tip_sign = tip_loc >= 0
                
                #   if thumb tip is on the opposite side of dividing plane as middle mcp, count as up
                if tip_sign != m_mcp_sign or tip_loc == 0:
                    count += 1
                
                # set value in game_position
                self.game_position[player][hand] = count
    
    def close_landmarker(self):
        '''
        Closes the hand landmarker object.
        '''
        self.landmarker.close()
        
    def draw_on_image(self, rgb_image, type="landmarks", detection_result: mp.tasks.vision.HandLandmarkerResult=None):
        '''
        Draws the requested annotation type onto the image and returns it.
        Includes a modified version of the code from Google's Mediapipe example 
        (https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
        '''
        # use instance result if none supplied 
        # or return unchanged image if no landmarks found
        if not detection_result:
            if not self.result:
                return rgb_image
            detection_result = self.result
        
        match(type):
            case "landmarks":
                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness
                annotated_image = np.copy(rgb_image)

                # Loop through the detected hands to visualize.
                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]
                    handedness = handedness_list[idx]

                    # Draw the hand landmarks.
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

                    # Get the top left corner of the detected hand's bounding box.
                    height, width, _ = annotated_image.shape
                    x_coordinates = [landmark.x for landmark in hand_landmarks]
                    y_coordinates = [landmark.y for landmark in hand_landmarks]
                    text_x = int(min(x_coordinates) * width)
                    text_y = int(min(y_coordinates) * height) - Hand.MARGIN

                    # Draw handedness (left or right hand) on the image.
                    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                                Hand.FONT_SIZE, Hand.HANDEDNESS_TEXT_COLOR, Hand.FONT_THICKNESS, cv2.LINE_AA)
            case _:
                pass
        return annotated_image