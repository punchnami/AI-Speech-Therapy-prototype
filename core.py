import os
import json
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, AutoModel
from huggingface_hub import InferenceClient
from evaluate import load
from groq import Groq
import random
from huggingface_hub import login

os.environ["PATH"] += os.pathsep + "C:/ffmpeg/bin" 


API_TOKEN = "HUGGINGFACE_API_KEY"

login(token=API_TOKEN)
model = AutoModel.from_pretrained("openai/whisper-small", token=API_TOKEN)
# processor = AutoProcessor.from_pretrained("stringbot/whisper-small-hi")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("stringbot/whisper-small-hi")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None

def choose_word():
    with open("wordlib.json", "r") as file:
        data = json.load(file)

    data["words"].sort(key=lambda x: x["score"])
    return [word["word"] for word in data["words"]]

def generate():
    words = choose_word()
    ran = random.randint(0, 9)
    w = words[ran]
    return call(w)

def call(w):

    client = Groq(
        api_key = "YOUR_API_KEY",
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "you are a speech therapist for a child of age 6."
            },
            {
                "role": "user",
                "content": f"Please generate one short sentence for a child of age 6 to read with the following word: {w}",
            }
        ],

        # The language model which will generate the completion.
        model="llama-3.1-8b-instant",
        stop=None,
        stream=False,
    )

    # Print the completion returned by the LLM.
    return chat_completion.choices[0].message.content

def transcription_func(audio_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
    
    transcription = asr_pipeline(audio_data)
    print("Transcription:", transcription["text"])
            
    return transcription["text"]

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

def evaluation(transcription, ref):
    # Create new testing dataset/references
    # user_input =  [real_time_transcription(press_record=True)]
    user_input = [transcription]
    references = [ref]
    
    wer = load("wer")
    score = 100 * (1-wer.compute(references=references, predictions=user_input))
    
    return score

if __name__ == '__main__':
    print('score: ',evaluation())
