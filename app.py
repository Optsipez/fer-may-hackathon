import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='emotion_classification_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess frames
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    normalized = rgb_frame / 255.0
    reshaped = np.expand_dims(normalized, axis=0)
    return reshaped.astype(np.float32)

# Streamlit app
st.title("Facial Emotion Detection")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    cap = cv2.VideoCapture(video_bytes)

    stframe = st.empty()
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

        # Run inference
        interpreter.invoke()

        # Get the output tensor and post-process
        output_data = interpreter.get_tensor(output_details[0]['index'])
        emotion = emotion_labels[np.argmax(output_data)]

        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        stframe.image(frame, channels="BGR")

        processed_frames += 1
        progress_bar.progress(processed_frames / frame_count)

    cap.release()
    progress_bar.empty()
