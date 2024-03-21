import numpy as np
import tensorflow as tf
import librosa
import gradio as gr

# Load your emotion recognition model
model = tf.keras.models.load_model('cnnmodel.h5')

def extract_mfcc(wav_file_name):
    # This function extracts mfcc features and obtains the mean of each dimension
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict_emotion(audio_file):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(audio_file.name)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]

# Create a Gradio interface
audio_input = gr.inputs.Audio(label="Upload your audio file")
output_text = gr.outputs.Textbox(label="Predicted Emotion")

gr.Interface(fn=predict_emotion, inputs=audio_input, outputs=output_text, title="Emotion Recognition").launch()
