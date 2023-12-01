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
or by installing the packages individually as listed above.


A working camera should also be available for the script to use. The camera should be oriented such that both players' hands are in frame, and the background should have an appropriate amount of contrast with both players' hands. The players should face towards each other where player one is on the bottom-right of the video stream and player two is on the top-left of the video stream.

# Usage
Run `__main__.py` to start the program. If needed, change the variable `VIDEO_INPUT_INDEX` in `__main__.py` to a different integer to use a different camera (by default, the camera used is the first camera device).

Once the video stream is shown, both players should move their hands into frame (a total of 4 hands). The count for each hand and the assigned player will be overlayed on the center of each hand. Note that hands that are not assigned to players are assumed to be zero (ie. if only player one has hands in frame, player two's hands are assumed to both be zero. No suggested moves are made in this example since the game position is in an end state where no moves can be made).

Suggestions will be printed to the console. They are in the format `move move_parameter` where `move` is either `tap` or `split`.
* For `tap`, `move_parameter` is the hand used by the moving player and the targeted opponent's hand. For instance, `tap l r` would mean the moving player should use their left hand to attack their opponent's right hand.
* For `split`, `move_parameter` is the desired count on the moving player's right hand after the split. For instance, `split 2` would mean the moving player should split their hands such that the right hand ends up with a count of 2.

To exit the program, press `q`.

## Rules
When calculating the next best move, the algorithm uses the default rules plus the following [variations](https://en.wikipedia.org/wiki/Chopsticks_(hand_game)#Variations):
* Hands are only eliminated when it has exactly 5 points, otherwise perform modular arithmetic (modulo 5).
* When splitting, players are allowed to eliminate one hand.
* When splitting, players can rollover via modular arithmetic (modulo 5).

# Additional Usages
In the Tests folder, there are scripts which use the `hand` and `algorithm` module for the following:
* `algo.py`: Provides the next best move given a single game position input
* `hands_test.py`: Shows hand landmark locations and game position using OpenCV and MediaPipe
* `sim_game_pve.py`: A simulated console based game of chopsticks played against the computer
* `sim_game.py`: A simulated console based game of chopsticks played against another player within the same console interface

# Additional Resources
* [Wikipedia: Chopsticks (hand game)](https://en.wikipedia.org/wiki/Chopsticks_(hand_game))
* [MediaPipe: Hand Landmarking](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python)
* [MediaPipe: Hand Landmarking Python Example](https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb)
* [Wikipedia: Negamax Algorithm](https://en.wikipedia.org/wiki/Negamax)
