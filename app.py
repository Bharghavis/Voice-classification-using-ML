import streamlit as st
import numpy as np    
import tensorflow as tf
import os
import urllib
import librosa

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition','View Source Code')
    )
            
    if selected_box == 'Emotion Recognition':        
        st.sidebar.success('Try by uploading an audio file.')
        application()
    elif selected_box == 'View Source Code':
        st.code(get_file_content_as_string("app.py"))

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/Bharghavis/Voice-classification-using-ML/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@st.cache(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('mymodel.h5')
    return model

def application():
    st.write("Loading models...")
    model = load_model()
    st.write("Models loaded successfully!")

    file_to_be_uploaded = st.file_uploader("Choose an audio file...", type="wav")
    
    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')
        st.write('Emotion of the audio is:', predict(model, file_to_be_uploaded))

def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict(model, wav_file):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_file)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]

if __name__ == "__main__":
    main()

