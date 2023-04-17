# Facial Animation: A Real-Time 3D Face Mapping


## Group Members
- Diven Ashwani Ahuja dahuja1@uncc.edu
- Saumit Chinchkhandi schinchk@uncc.edu
- Sayyam Anil Gada sgada@uncc.edu
- Tushar Nemade tnemade@uncc.edu

## Project Introduction
Mediapipe is a powerful machine learning framework developed by Google, which provides an extensive set of pre-built tools and models for various computer vision and machine learning tasks. One of the most widely used features of Mediapipe is its face detection and tracking module, which can detect and track facial features in real-time with high accuracy and robustness.

In this project, we aim to leverage the power of Mediapipe's face detection and tracking module to build an application that can detect and track faces using live camera. We plan to use the Mediapipe ML model to visualize the detected faces and identify facial boundaries and feautures.

The primary goal of this project is to provide a user-friendly and intuitive interface for real-time face detection and tracking, which can be used for various applications such as video analytics, security systems, augmented reality, and more. By using Mediapipe's robust and efficient face detection and tracking algorithm, we hope to achieve high accuracy and performance while maintaining a low computational footprint.


## ML Model Insight
Using MediaPipe Face Mesh, 468 3D face landmarks may be calculated in real-time. With only a single camera input, it employs machine learning (ML) to infer the 3D facial surface. The strategy provides the real-time throughput critical for live experiences by utilizing lightweight model architectures throughout the pipeline.

It uses transfer learning to train a model that concurrently predicts 2D semantic contours on annotated real-world data and 3D landmark coordinates on synthetic rendered data for 3D facial landmarks. It generates 3D landmark predictions from the resulting model using both synthetic and real-world data.

A clipped video frame without any additional depth input is passed to the 3D landmark network as input. The 3D point coordinates and detects the likelihood that a face is recognized and adequately aligned in the input are both output by the model. Predicting a 2D heatmap for each landmark is a frequent alternative strategy, but it is not suitable for depth prediction and has high computational demands for such a large number of points. By iteratively bootstrapping and adjusting predictions, the accuracy and stability of the model are boosted.

![image](https://user-images.githubusercontent.com/115678929/220990722-33152c18-4527-47be-8bb9-e4537a33f47c.png)

## WebRTC
Real-time communication between web browsers and mobile applications is made possible via the open-source WebRTC (Web Real-Time Communication) technology, which was created by Google. It enables the direct integration of audio and video communication into web pages without the need for any additional plugins or applications. To offer a smooth real-time communication experience, WebRTC employs common web technologies including HTML, CSS, and JavaScript. In order to facilitate applications like video conferencing, live streaming, and online gaming, it supports peer-to-peer connection between browsers or mobile devices. Several operating systems, including Windows, MacOS, Linux, Android, and iOS, as well as popular web browsers including Chrome, Firefox, Safari, and Edge support WebRTC.

## Github Model Reference

[MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh)

## Architecture
   
   ### Client-Server Architecture

   The Client-Server model with WebRTC and ML model describes a system where clients use WebRTC technology to establish a real-time, peer-to-peer communication channel with a server. The server hosts a machine learning model that is used to process and analyze data received from the clients. The clients can send data to the server in real-time, and receive results from the ML model, allowing for dynamic and interactive communication between the two components. This setup enables users to receive immediate results from the ML model without the need for a long and slow data transfer process, making it a highly efficient and effective solution for a wide range of applications.

   ![Client Server Model](https://github.com/tushar251095/CCN_Human_3D_Modeling/blob/main/Architecture.jpg?raw=true)
   
   
## Project Progress 1
In Project Progress 1, we have completed Week 1-2 iterations as mentioned in our Project Plan. We have understood the funtionality of our chosen MediaPipe ML model where we have researched to gain deep insights about the working of this model and downloaded the required code. Additionally, we have fine tuned and cleaned the code to remove unnecessary functionalities and redesigned it based on our project requirements. We later created a virtual environment for installing any dependencies and executing our development.

While working on the code, our team understood various parameters associated which contribute to the outcome of the model during runtime like max_num_faces which denotes the maximum number of faces to detect, min_detection_confidence and min_tracking_confidence which helps configure face detection success rates. The model as of now runs as a windows application only.

For working on future project iterations, our team is currently learning Streamlit to create the frontend for our Face Mesh Application. Additionally, our team is working on providing the User run time options like changing number of faces to be detected and a slider to set ratio of face detection and tracking confidence.

## Project Progress 1 Setup

Below mentioned are steps required to run the python file:-
1. Create and initialize virtual environment [Link](https://docs.python.org/3/library/venv.html)
2. Upgrade pip. ```python -m pip install --upgrade pip```
3. Upgrade pip dependencies. ```pip install --upgrade pip setuptools wheel```
4. Install all the dependencies for the project. ```pip install streamlit mediapipe streamlit_webrtc```
5. Execute the python file. ```python face_mesh.py```

Key Points to Note:
1. Mediapipe requires python version 3.9 or lower to run so in case there is a higher version installed, switch to the lower python versions in an environment.
2. To forcefully terminate the application, press ```ctrl+C```.

## Project Progress 2
In Project Progress 2, our team has successfully completed developing and integrating Streamlit into our project. In previous iteration, we ran pre-trained machine learning model in windows application, but now with Streamlit, we have seamlessly designed a responsive Frontend for our Face Mesh detection using MediaPipe. Additionally, Streamlit has been extensively used to develop our backend server for helping us implement a single server multiple clients framework for our project.

In the Frontend application developed, we have given user options to specify the number of faces to be detected while using Webcam. Min_Tracking_Confidence and Min_Detection_confidence bars which range from 0 to 100 which are seekbars which can be adjusted to modify face detection and identification rates. These Key Performance Indicators(KPI's) are essential in tweaking results based on different parameters. During execution, the server is started which can be accessed based on a shareable link which is sent to multiple devices. On opening the link, Webcam/Camera access needs to be granted and the application soons starts detecting faces visible through the device camera.

For future iterations, our team is wokring towards understanding how WebRTC can be used to add real-time communication capabilities to our Face Mapping project.

## Steps To Execute Progress 2 development:-
1. Create and initialize virtual environment [Link](https://docs.python.org/3/library/venv.html)
2. Upgrade pip. ```python -m pip install --upgrade pip```
3. Upgrade pip dependencies. ```pip install --upgrade pip setuptools wheel```
4. Install all the dependencies for the project if not installed. ```pip install -r requirements.txt```
5. Execute the python file. ```streamlit run face_mesh.py```

## Project Progress 3
In the iteration 2, we had completed integrating streamlit into our face detection project. In the current iteration, we have converted streamlit into streamlit webrtc for implementing webrtc into streamlit. The main project goal was to implement real-time communication while running the project, which is crucial because it allows users to quickly verify what their models can do with handy video input from their own devices, such as webcams or smartphones. This key functionality was achieved with streamlit webrtc. Webrtc uses WebRTC ice server which is a Session Traversal Utilities for NAT(STUN) server which is used for relaying data.

## Steps To Execute Progress 2 development:-
1. Create and initialize virtual environment [Link](https://docs.python.org/3/library/venv.html)
2. Upgrade pip. ```python -m pip install --upgrade pip```
3. Upgrade pip dependencies. ```pip install --upgrade pip setuptools wheel```
4. Install all the dependencies for the project if not installed. ```pip install -r requirements.txt```
5. Execute the python file. ```streamlit run face_mesh.py```


## Project plan

| Week | Plan |
|----------|----------|
| Week 1-2 | Setting up Project Environment and Downloading and understanding mediapipe face mesh and its usage for the project |
| Week 3-4 | Design of Client-Server Architecture and implementation of WebRTC protocol to process media between devices |
| Week 5-6 | Integration of WebRTC and ML model with Client-Server Architecture |
| Week 7-8 | Conducting rigorous Testing and Optimization phases for achieving maximum throughput for the developed application and implementing Project Report and PowerPoint Presentation for documenting project findings and conclusion |
