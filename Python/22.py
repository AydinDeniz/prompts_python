
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load an audio file
def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

# Visualize spectrogram
def plot_spectrogram(y, sr):
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.show()

# Noise reduction
def reduce_noise(y, sr):
    noise = np.random.normal(0, 0.005, y.shape)
    return y + noise

# Pitch shifting
def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)

# Load dataset and extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0)

# Prepare dataset
def load_dataset(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file)
                feature = extract_features(file_path)
                X.append(feature)
                y.append(label)
    return np.array(X), np.array(y)

# Create and train a neural network model
def train_audio_classifier(data_dir):
    X, y = load_dataset(data_dir)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    
    print("Model trained successfully")
    model.save("audio_classifier.h5")

if __name__ == "__main__":
    # Example Usage
    file_path = "example.wav"
    y, sr = load_audio(file_path)
    
    print("Plotting Spectrogram...")
    plot_spectrogram(y, sr)

    print("Applying Noise Reduction...")
    y_denoised = reduce_noise(y, sr)

    print("Applying Pitch Shifting...")
    y_shifted = pitch_shift(y, sr)

    print("Training Audio Classifier...")
    train_audio_classifier("audio_dataset")
