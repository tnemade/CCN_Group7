import streamlit as st
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

stframe = st.empty()

vid = cv2.VideoCapture(0)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            continue

        results = face_mesh.process(frame)
        frame.flags.writeable = True

        face_count = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
        stframe.image(frame, channels="BGR", use_column_width=True)