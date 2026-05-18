# train.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam

import pickle
import os

# Check if preprocessed data exists
if not os.path.exists('preprocessed_data.pkl'):
    print("Error: preprocessed_data.pkl not found.")
    print("Run: python preprocess.py")
    exit(1)

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    network_input, network_output, pitchnames = pickle.load(f)

print("Data loaded successfully!")

print(f"Input shape: {network_input.shape}")
print(f"Output shape: {network_output.shape}")

# Build model
model = Sequential()

model.add(Input(
    shape=(network_input.shape[1], network_input.shape[2])
))

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256))
model.add(Dropout(0.3))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(len(pitchnames), activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001)
)

# Model summary
model.summary()

# Train model
model.fit(
    network_input,
    network_output,
    epochs=50,
    batch_size=64
)

# Save model
model.save('model.keras')

print("Model training completed!")
print("Model saved as model.keras")