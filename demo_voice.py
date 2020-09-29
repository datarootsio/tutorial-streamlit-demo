import streamlit as st
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import sounddevice as sd
import wavio
import glob
from helper import draw_embed, create_spectrogram, read_audio


st.title("Streamlit showcase")

model_load_state = st.text("Loading pretrained models...")

seed = 42
low_mem = False
num_generated = 0
enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
syn_model_dir = Path("synthesizer/saved_models/logs-pretrained/")
voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")
encoder.load_model(enc_model_fpath)
synthesizer = Synthesizer(
    syn_model_dir.joinpath("taco_pretrained"), low_mem=low_mem, seed=seed
)
vocoder.load_model(voc_model_fpath)

model_load_state.text("Loaded pretrained models!")

st.subheader("1. Record your own voice")


if st.button("Click to Record"):
    record_state = st.text("Recording...")

    duration = 5  # seconds
    fs = 48000
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    record_state.text("Saving sample as myvoice.mp3")
    path_myrecording = "./samples/myvoice.mp3"
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    st.audio(read_audio(path_myrecording))
    record_state.text("Done! Saved sample as myvoice.mp3")

    fig = create_spectrogram(path_myrecording)
    st.pyplot(fig)

st.subheader("2. Choose an audio record")


audio_folder = "samples"
filenames = glob.glob(os.path.join(audio_folder, "*.mp3"))
selected_filename = st.selectbox("Select a file", filenames)
in_fpath = Path(selected_filename.replace('"', "").replace("'", ""))

if in_fpath is not None:
    # - If the wav is already loaded:
    original_wav, sampling_rate = librosa.load(str(in_fpath))
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    st.write("Loaded file succesfully!")
    embed = encoder.embed_utterance(preprocessed_wav)
    st.audio(read_audio(in_fpath))
    st.success("Created the embedding")

    fig = draw_embed(embed, "myembedding", None)
    st.pyplot(fig)


st.subheader("3. Synthesize text.")
text = st.text_input("Write a sentence (+-20 words) to be synthesized:")

# The synthesizer works in batch, so you need to put your data
# in a list or numpy array
if text != "":
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are,
    # you can retrieve them here by passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    st.write("Created the mel spectrogram")

    # Generating the waveform
    st.write("Synthesizing the waveform:")

    generated_wav = vocoder.infer_waveform(spec)

    # Post-generation
    # There's a bug with sounddevice that makes the audio cut one
    # second earlier, so we pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)

    # Play the audio (non-blocking)
    try:
        sd.stop()
        sd.play(generated_wav, synthesizer.sample_rate)
    except sd.PortAudioError as e:
        st.write("\nCaught exception: %s" % repr(e))
        st.write(
            'Continuing without audio playback. Suppress this message with \
            the "--no_sound" flag.\n'
        )

    # Save it on the disk
    filename = "demo_output_%02d.wav" % num_generated
    st.write(generated_wav.dtype)
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    num_generated += 1
    st.write("\nSaved output as %s\n\n" % filename)
