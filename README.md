🎵 AI Music Generation using LSTM

This project is an AI-based music generator that uses an LSTM neural network to learn musical patterns from MIDI files and generate new melodies. The model is already trained, and all required data is accessed automatically via a pre-configured Google Drive link inside the generator script.

📁 Repository

GitHub:
https://github.com/Rakeentheboss/CodeALpha_music_generation

⚙️ Project Overview

The system works in the following way:

MIDI files are converted into note sequences during preprocessing.
A sliding window approach is used where:
Input: 100 sequential notes
Output: Next predicted note
These sequences are fed into an LSTM network.
The LSTM processes each note step-by-step and builds a 256-dimensional internal representation of the musical context.
This representation captures the melody structure over time.
A Dense layer maps these learned features into actual pitch probabilities.
The model then predicts the next note in the sequence, gradually generating a full melody.
🧠 Model Architecture Insight
Input sequence length: 100 notes
LSTM units: 256
The model learns temporal dependencies by passing hidden state h_t through each timestep.
Each note contributes to updating the internal state, allowing the model to “remember” the melody structure.
Final Dense layer converts the learned representation into a probability distribution over possible notes.
The output is then sampled to generate new music sequences.
📦 Installation

Clone the repository:

git clone https://github.com/Rakeentheboss/CodeALpha_music_generation
cd CodeALpha_music_generation

Install dependencies:

pip install -r requirements.txt
🚀 How to Run
▶️ Run the Music Generator (Streamlit App)

Simply execute:

streamlit run generate.py

This will launch a local web interface where you can generate music.

🔗 Data Handling (Important)
The dataset and model dependencies are already handled automatically
A Google Drive link is hardcoded inside generate.py
Therefore:

✔ No need to manually download dataset
✔ No need to run preprocessing
✔ No need to retrain the model

🧪 Optional: Preprocessing Script

A preprocessing script (preprocess.py) is included for reference:

python preprocess.py

However, it is NOT required to run the project, since:

Preprocessed data is already stored in Google Drive
The generator directly accesses it via the embedded link
🎼 Output
Generated output is converted back from predicted note sequences into MIDI format
Users can listen to the generated melody directly from the Streamlit interface
Output files can be downloaded instantly
📌 Tech Stack
Python
TensorFlow / Keras
LSTM Neural Network
Streamlit
Music21 (for MIDI processing)
📜 License

This project is for educational and research purposes.
