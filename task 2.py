import streamlit as st
import torch
import torchaudio
import io
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set audio backend (optional but safe for Windows)
torchaudio.set_audio_backend("soundfile")

# Load model & processor
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    return processor, model

processor, model = load_model()

# Transcription function
def transcribe_audio(uploaded_file):
    audio_bytes = uploaded_file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load waveform
    waveform, sample_rate = torchaudio.load(audio_buffer)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Preprocess and infer
    inputs = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    return processor.batch_decode(predicted_ids)[0].lower()

# Streamlit UI
st.title("🎙️ Wav2Vec2 Speech-to-Text")
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.info("Transcribing...")
    try:
        text = transcribe_audio(uploaded_file)
        st.success("Transcription:")
        st.write(text)
    except Exception as e:
        st.error(f"❌ Error: {e}")
