import streamlit as st
from core import generate
# import sounddevice as sd
import pyaudio
import numpy as np
import wave
import core 

st.title("🎤 Can you get 5 stars on this level?")

text = generate()
st.subheader(f"Say: {text}")

def record_audio():
    st.write("🎙️ Let's Practice!")

    SAMPLE_RATE = 16000  
    CHANNELS = 1  
    FORMAT = pyaudio.paInt16  
    CHUNK = 1024  
    DURATION = 5  

    audio = pyaudio.PyAudio()

    # Open the stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=CHUNK)

    st.write("🎤 Recording...")
    frames = []

    for _ in range(0, int(SAMPLE_RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("⏹️ Recording finished!")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save as WAV file
    file_path = "recordings/recorded_audio.wav"
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    st.success("✅ Recording saved!")
    return file_path

if st.button("Start Recording"):
    audio_file = record_audio()
    st.audio(audio_file, format="audio/wav")

    transcription = core.transcription_func(audio_file)
    st.subheader("📝 Transcription:")
    st.write(transcription)

    wer_score = core.evaluation(transcription, text)
    st.subheader("📊 Your Score:")
    # st.write(f"{wer_score:.2f}%")
    
    if(wer_score > 80):
        st.write(f"Great Job! ⭐⭐⭐⭐⭐")
    elif(wer_score > 60):
        st.write(f"Nice Job! ⭐⭐⭐⭐")
    elif(wer_score > 40):
        st.write(f"Nice Job! ⭐⭐⭐")
    elif(wer_score > 20):
        st.write(f"Nice Job! ⭐⭐")
    else:
        st.write(f"Nice Job! ⭐")
