# Sticks Suggestions
Final Project by Team CCL for ME369P: Application Programming for Engineers.

# Overview
stick-suggestions is a python program that provides the optimal next move for a game of [chopsticks (AKA sticks or calculator)](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)) in real time. It uses OpenCV and MediaPipe to detect the currect game position based on camera input.

# Prerequisites
The following packages should be installed before running the program:
* OpenCV: `opencv-python`
* MediaPipe: `mediapipe`
* NumPy: `numpy`

You can do so by pip installing the requirements.txt file:
```python
pip install requirements.txt
```

A working camera should also be available for the script to use. The camera should be oriented such that both players' hands are in frame and the background should have an appropriate amount of contrast with both players' hands. The players should face towards each other where player one is on the bottom or right of the video stream.

# Usage
Run `__main__.py` to start the program. 

Once the video stream is shown, both players should move their hands into frame. The count for each hand and the assigned player will be overlayed on the center of each hand. Suggestions will be printed to the console.

## Rules
When calculating the next best move, the algorithm uses the default rules plus the following [variations](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)#Variations):
* Hands are only eliminated when it has exactly 5 points, otherwise perform modular arithmitic (modulo 5).
* When splitting, players are allowed to kill off one hand.
* When splitting, players can rollover via modular arithmitic (modulo 5).

# Additional Resources
* [Wikipedia: Chopsticks (hand game)](https://en.wikipedia.org/wiki/Chopsticks_(hand_game))
* [MediaPipe: Hand Landmarking](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python)
* [MediaPipe: Hand Landmarking Python Example](https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
* [Wikipedia: Negamax Algorithm](https://en.wikipedia.org/wiki/Negamax)