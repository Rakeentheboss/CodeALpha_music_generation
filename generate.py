from flask import Flask, request, send_file
import numpy as np
import pickle
import random
import tempfile
import gdown
import os

from tensorflow.keras.models import load_model
from music21 import note, chord, stream, instrument

app = Flask(__name__)

FILE_ID = "1ueu_XEgrF5Sekpd8fbGMewFHTR0-tXGJ"
PKL_PATH = "preprocessed_data.pkl"

model = load_model("model.keras")


def load_data():
    if not os.path.exists(PKL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        gdown.download(url, PKL_PATH, quiet=False)

    with open(PKL_PATH, "rb") as f:
        network_input, _, pitchnames = pickle.load(f)

    int_to_note = {i: n for i, n in enumerate(pitchnames)}
    return network_input, pitchnames, int_to_note


network_input, pitchnames, int_to_note = load_data()


@app.route("/generate")
def generate():
    num_notes = int(request.args.get("notes", 200))

    sequence_length = len(network_input[0])
    pattern = np.array(random.choice(network_input))

    output_notes = []
    offset = 0

    for _ in range(num_notes):
        input_seq = np.reshape(pattern, (1, sequence_length, 1))
        input_seq = input_seq / float(len(pitchnames))

        pred = model.predict(input_seq, verbose=0)[0]
        pred = np.log(pred + 1e-8) / 0.8
        pred = np.exp(pred) / np.sum(np.exp(pred))

        index = np.random.choice(len(pred), p=pred)

        result = int_to_note[index]
        output_notes.append(result)

        pattern = np.append(pattern, index)[-sequence_length:]

    midi_stream = stream.Stream()

    for item in output_notes:
        if "." in item or item.isdigit():
            notes = item.split(".")
            chord_notes = [note.Note(int(n)) for n in notes]
            midi_stream.append(chord.Chord(chord_notes))
        else:
            midi_stream.append(note.Note(item))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    midi_stream.write("midi", fp=tmp.name)

    return send_file(tmp.name, as_attachment=True, download_name="music.mid")


if __name__ == "__main__":
    app.run(debug=True)