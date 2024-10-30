# Try On Glasses Web App

A Flask web application that allows users to try on virtual glasses in real-time through their webcam feed. The application detects the user's face and overlays different glasses filters. Users can navigate between glasses options and capture images by interacting with virtual buttons displayed on the video feed.

## Features

- **Face Detection**: Uses OpenCV's Haar Cascade to detect faces in the webcam feed.
- **Glasses Overlay**: Overlays virtual glasses on the detected face.
- **Hand Detection with Buttons**: Uses Mediapipe to detect hand gestures and interact with virtual buttons for:
  - Switching to the next pair of glasses.
  - Capturing the current frame with glasses overlay.
- **Flask Web App**: Runs the application in a web server using Flask, accessible through a browser.

