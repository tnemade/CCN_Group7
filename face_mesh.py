import streamlit as st
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

from streamlit_webrtc import webrtc_streamer
import av



st.title("Face Mesh")
max_faces = st.number_input("Maximum Number of Faces", value = 2, min_value=1)
st.markdown("---")
detection_confidence = st.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.5)
tracking_confidence = st.slider("Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.5)
st.markdown("---")



def video_frame_callback(frame):
    frame = frame.to_ndarray(format="bgr24")

    st.markdown("## Output")
   
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    kpi1_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)


    with mp_face_mesh.FaceMesh(
        max_num_faces = max_faces,
        min_detection_confidence = detection_confidence,
        min_tracking_confidence = tracking_confidence
    ) as face_mesh:

        results = face_mesh.process(frame)
        frame.flags.writeable = True

        face_count = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_count +=1

                mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec
            )


        kpi1_text.write(f'<h1 style="text-align: center; color:red;">{face_count}</h1>', unsafe_allow_html=True)
        
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})