import streamlit as st
import cv2
import mediapipe as mp
import util
import pyautogui
import numpy as np
import tempfile
import import_ipynb
import project

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

st.title("Virtual Mouse Using Hand Gesture")
st.write("Press 'Start' to activate the virtual mouse.")

# Button to start webcam
if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for video frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        if processed.multi_hand_landmarks:
            for hand_landmarks in processed.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                project.detect_gestures(frame, landmarks_list,processed)

        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

