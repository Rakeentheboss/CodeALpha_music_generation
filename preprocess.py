# preprocess.py

from music21 import converter, instrument, note, chord
import glob
import numpy
import pickle
from tensorflow.keras.utils import to_categorical

# Load MIDI files
files = glob.glob("dataset/*.midi")
print(f"Found {len(files)} MIDI files")

notes = []

# Extract notes/chords
for file in files:
    try:
        print(f"Processing: {file}")

        midi = converter.parse(file)
        parts = instrument.partitionByInstrument(midi)

        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flatten().notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    except Exception as e:
        print(f"Error processing {file}: {e}")

print(f"\nTotal notes extracted: {len(notes)}")

# OPTIONAL:
# Use smaller dataset first for testing
notes = notes[:100000]

# Sequence length
sequence_length = 100

# Unique pitch names
pitchnames = sorted(set(notes))

# Note -> Integer mapping
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

# Create sequences
for i in range(0, len(notes) - sequence_length):

    sequence_input = notes[i:i + sequence_length]
    sequence_output = notes[i + sequence_length]

    network_input.append([note_to_int[char] for char in sequence_input])
    network_output.append(note_to_int[sequence_output])

n_patterns = len(network_input)

print(f"Total patterns: {n_patterns}")

# Reshape input
network_input = numpy.reshape(
    network_input,
    (n_patterns, sequence_length, 1)
)

# Normalize input
network_input = network_input / float(len(pitchnames))

# One-hot encode output
network_output = to_categorical(
    network_output,
    num_classes=len(pitchnames)
)

# Save preprocessed data
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((network_input, network_output, pitchnames), f)

print("Preprocessing completed successfully!")