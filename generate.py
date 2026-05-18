import streamlit as st
import random
import numpy as np
import os
import tempfile
import pickle
import gdown

from tensorflow.keras.models import load_model
from music21 import instrument, note, chord, stream


# =========================
# CONFIG
# =========================
FILE_ID = "1ueu_XEgrF5Sekpd8fbGMewFHTR0-tXGJ"
PKL_PATH = "preprocessed_data.pkl"


# =========================
# LOAD MODEL (cached)
# =========================
@st.cache_resource
def load_music_model(model_path):
    return load_model(model_path)


# =========================
# LOAD DATA (cached + Drive download)
# =========================
@st.cache_resource
def load_data():
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    if not os.path.exists(PKL_PATH):
        gdown.download(url, PKL_PATH, quiet=False)

    with open(PKL_PATH, "rb") as f:
        network_input, network_output, pitchnames = pickle.load(f)

    int_to_note = {i: n for i, n in enumerate(pitchnames)}
    return network_input, pitchnames, int_to_note


# =========================
# UI
# =========================
st.title("🎹 AI Music Generator (MIDI)")
st.write("LSTM-based music generation using Music21 + Streamlit")

model_path = st.text_input("Model path", "model.keras")

num_notes = st.slider("Notes to generate", 50, 500, 200)
temperature = st.slider("Temperature", 0.5, 1.5, 0.8)

generate = st.button("🎼 Generate MIDI")


# =========================
# MAIN LOGIC
# =========================
if generate:

    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()

    # Load resources
    model = load_music_model(model_path)
    network_input, pitchnames, int_to_note = load_data()

    sequence_length = len(network_input[0])

    # Seed selection
    seed_index = random.randint(0, len(network_input) - 1)
    pattern = np.array(network_input[seed_index])

    prediction_output = []

    st.info("Generating music...")

    # =========================
    # GENERATION LOOP
    # =========================
    for _ in range(num_notes):

        prediction_input = np.reshape(pattern, (1, sequence_length, 1))
        prediction_input = prediction_input / float(len(pitchnames))

        prediction = model.predict(prediction_input, verbose=0)[0]

        # Temperature sampling
        prediction = np.log(prediction + 1e-8) / temperature
        prediction = np.exp(prediction)
        prediction = prediction / np.sum(prediction)

        index = np.random.choice(len(prediction), p=prediction)

        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[-sequence_length:]  # FIXED WINDOW

    # =========================
    # CONVERT TO MIDI
    # =========================
    offset = 0
    output_notes = []

    for item in prediction_output:

        if '.' in item or item.isdigit():

            notes_in_chord = item.split('.')
            notes_list = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.AcousticGuitar()
                notes_list.append(new_note)

            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(item)
            new_note.offset = offset
            new_note.storedInstrument = instrument.AcousticGuitar()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    # =========================
    # SAVE MIDI
    # =========================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp:
        midi_path = tmp.name

    midi_stream.write('midi', fp=midi_path)

    with open(midi_path, "rb") as f:
        midi_bytes = f.read()

    st.success("MIDI generated successfully ")



    st.download_button(
        label=" Download MIDI",
        data=midi_bytes,
        file_name="generated_music.mid",
        mime="audio/midi"
    )