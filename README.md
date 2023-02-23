# SpeakUp: Real-Time 3D Face Mapping


## Group Members
- Diven Ashwani Ahuja dahuja1@uncc.edu
- Saumit Chinchkhandi schinchk@uncc.edu
- Sayyam Anil Gada sgada@uncc.edu
- Tushar Nemade tnemade@uncc.edu

## Introduction
Using MediaPipe Face Mesh, 468 3D face landmarks may be calculated in real-time. With only a single camera input, it employs machine learning (ML) to infer the 3D facial surface. The strategy provides the real-time throughput critical for live experiences by utilizing lightweight model architectures and GPU acceleration throughout the pipeline.
It uses transfer learning to train a model that concurrently predicts 2D semantic contours on annotated real-world data and 3D landmark coordinates on synthetic rendered data for 3D facial landmarks. It generates 3D landmark predictions from the resulting model using both synthetic and real-world data.
A clipped video frame without any additional depth input is passed to the 3D landmark network as input. The 3D point coordinates and detects the likelihood that a face is recognized and adequately aligned in the input are both output by the model. Predicting a 2D heatmap for each landmark is a frequent alternative strategy, but it is not suitable for depth prediction and has high computational demands for such a large number of points. By iteratively bootstrapping and adjusting predictions, the accuracy and stability of the model are boosted. By doing so, we may expand our dataset to include progressively difficult scenarios like grimaces, oblique angles, and occlusions.

## Github Model Reference

[MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh)

## Architecture
   
   ### Client-Server Architecture

   The Client-Server model with WebRTC and ML model describes a system where clients use WebRTC technology to establish a real-time, peer-to-peer communication channel with a server. The server hosts a machine learning model that is used to process and analyze data received from the clients. The clients can send data to the server in real-time, and receive results from the ML model, allowing for dynamic and interactive communication between the two components. This setup enables users to receive immediate results from the ML model without the need for a long and slow data transfer process, making it a highly efficient and effective solution for a wide range of applications.

   ![Client Server Model](https://github.com/tushar251095/CCN_Human_3D_Modeling/blob/main/Architecture.jpg?raw=true)

## Project plan

| Week | Plan |
|----------|----------|
| Week 1-2 | Download and train the MediaPipe Face Mesh model using the pre-processed datasets |
| Week 3-4 | Design of Client-Server Architecture and Implementation of WebRTC design |
| Week 5-6 | Integration of WebRTC and ML model with Client-Server Architecture |
| Week 7-8 | Testing and Optimization for the developed application and Project Report and PPT implementation |
