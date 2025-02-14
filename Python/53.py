
import os
import whisper
import openai
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import soundfile as sf

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Load Whisper model
whisper_model = whisper.load_model("base")

# Transcribe podcast audio
def transcribe_audio(audio_path):
    print("Transcribing podcast audio...")
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(transcript)
    return transcript

# Extract key topics using NLP
def extract_topics(text, num_topics=5):
    print("Extracting topics from podcast transcript...")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42)
    kmeans.fit(X)
    
    topic_sentences = []
    for i in range(num_topics):
        topic_index = np.argmax(kmeans.cluster_centers_[i])
        topic_sentences.append(sentences[topic_index])

    with open("topics.json", "w", encoding="utf-8") as f:
        json.dump(topic_sentences, f, indent=4)
    
    return topic_sentences

# Summarize transcript using GPT-4
def summarize_transcript(text):
    print("Generating podcast summary...")
    prompt = f"Summarize the following podcast transcript:
{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in summarizing long-form podcasts."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    summary = response["choices"][0]["message"]["content"]
    
    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    return summary

# Generate spectrogram visualization
def generate_spectrogram(audio_path):
    print("Generating spectrogram visualization...")
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Podcast Spectrogram")
    plt.savefig("spectrogram.png")
    plt.close()

# Split audio into segments
def split_audio(audio_path, segment_length=30):
    print("Splitting audio into segments...")
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    segments = []
    for start in range(0, int(duration), segment_length):
        end = min(start + segment_length, int(duration))
        segment = y[start * sr:end * sr]
        segment_filename = f"segment_{start}_{end}.wav"
        sf.write(segment_filename, segment, sr)
        segments.append(segment_filename)

    return segments

if __name__ == "__main__":
    podcast_audio = "podcast_episode.mp3"

    if not os.path.exists(podcast_audio):
        print("Podcast audio file not found!")
        exit()

    transcript = transcribe_audio(podcast_audio)
    topics = extract_topics(transcript)
    summary = summarize_transcript(transcript)
    generate_spectrogram(podcast_audio)
    segments = split_audio(podcast_audio)

    print("Podcast processing complete.")
    print("Extracted Topics:", topics)
    print("Summary saved to 'summary.txt'.")
    print("Spectrogram saved as 'spectrogram.png'.")
    print("Audio segments saved:", segments)
