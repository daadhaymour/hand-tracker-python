#import mediapipe (library that makes it happen!!)

import cv2
import mediapipe as mp

#drawing lines in-between the land marks/hand joints
mp_drawing = mp.solutions.drawing_utils

#variable if u want to style the interconnecting lines
mp_drawing_styles = mp.solutions.drawing_styles

#tracking hands
mphands = mp.solutions.hands

#access device webcam
cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    data, image = cap.read()
    #flip image
    image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
    #store results
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #i want screen to be smaller
    image = cv2.resize(image, (640,420))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mphands.HAND_CONNECTIONS
            )
    
    cv2.imshow('Hand Tracker', image)
    cv2.waitKey(1)
