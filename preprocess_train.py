import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
import pickle
import os

# --- Configuration ---
MAX_INPUT_LEN = 20
MAX_TARGET_LEN = 20
EMBEDDING_DIM = 100
LSTM_UNITS = 128
BATCH_SIZE = 10
EPOCHS = 500
VALIDATION_SPLIT = 0.2

# --- File Paths ---
DATA_PATH = 'data/chat_data.txt'
TOKENIZER_PATH = 'data/tokenizer.pkl'
PROCESSED_DATA_PATH = 'data/chat_data.pkl'
MODEL_SAVE_PATH = 'chatbot_seq2seq_model.keras'

# --- Load and preprocess data ---
def load_and_preprocess_data(file_path):
    questions, answers = [], []
    skipped_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().split('\n')

        print(f"Total lines read from file: {len(lines)}")

        for i, line in enumerate(lines):

            processed_line = line.strip()


            if not processed_line:
                skipped_count += 1

                continue

            # Attempt to split by tab
            pair = processed_line.split('\t')

            # --- CRITICAL FILTERING LOGIC ---

            if len(pair) == 2 and pair[0].strip() and pair[1].strip():
                questions.append(pair[0].strip())
                answers.append(f"start {pair[1].strip()} end")
            else:

                skipped_count += 1

                print(f"WARNING: Skipping line {i+1} due to formatting: '{line.strip()}'")
                print(f"         - Reason: Expected 'QUESTION\\tANSWER'. Found {len(pair)} parts and/or empty part(s).")
                if len(pair) == 2:
                    print(f"           Q_empty: {not bool(pair[0].strip())}, A_empty: {not bool(pair[1].strip())}")
                elif len(pair) > 2:
                    print(f"           - Too many tabs or unexpected data: {pair}")
                else:
                    print(f"           - No tab delimiter found or only one part: {pair}")


    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        return [], []
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return [], []

    print(f"\n--- Data Loading Summary ---")
    print(f" Loaded {len(questions)} conversation pairs.")
    print(f" Skipped {skipped_count} lines due to formatting issues or being empty.")
    print(f"Total lines processed (Loaded + Skipped): {len(questions) + skipped_count}")
    print(f"--------------------------\n")
    return questions, answers

# --- Tokenize and sequence ---
def tokenize_and_sequence(questions, answers):
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(questions + answers)
    vocab_size = len(tokenizer.word_index) + 1

    question_sequences = tokenizer.texts_to_sequences(questions)
    answer_sequences = tokenizer.texts_to_sequences(answers)

    encoder_input_data = pad_sequences(question_sequences, maxlen=MAX_INPUT_LEN, padding='post')
    decoder_input_data = pad_sequences(answer_sequences, maxlen=MAX_TARGET_LEN, padding='post')

    decoder_target_data = np.zeros_like(decoder_input_data)
    # Shift targets by one for teacher forcing
    decoder_target_data[:, :-1] = decoder_input_data[:, 1:]

    return tokenizer, vocab_size, encoder_input_data, decoder_input_data, decoder_target_data

# --- Save tokenizer and processed data ---
def save_data(tokenizer, encoder_input_data, decoder_input_data, decoder_target_data):
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)

    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROCESSED_DATA_PATH, 'wb') as f:
        pickle.dump({
            'encoder_input_data': encoder_input_data,
            'decoder_input_data': decoder_input_data,
            'decoder_target_data': decoder_target_data
        }, f)
    print("✅ Tokenizer and processed data saved.")

# --- Model creation ---
def create_seq2seq_chatbot_model(vocab_size, embedding_dim, lstm_units):
    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Training function ---
def train_model(vocab_size, encoder_input_data, decoder_input_data, decoder_target_data):
    model = create_seq2seq_chatbot_model(vocab_size, EMBEDDING_DIM, LSTM_UNITS)
    model.summary()

    print(" Training model...")

    if len(encoder_input_data) == 0:
        print("ERROR: No valid data found for training after preprocessing. Aborting training.")
        return

    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    model.save(MODEL_SAVE_PATH)
    print(f" Model saved to '{MODEL_SAVE_PATH}'")

    # Ensure the 'data' directory exists for history
    os.makedirs(os.path.dirname('data/training_history.pkl'), exist_ok=True)
    with open('data/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("📊 Training history saved.")

# --- Main pipeline ---
def main():
    # --- Step 1: Load and Preprocess Data ---
    print("\n--- Step 1: Loading and Preprocessing Data ---")
    questions, answers = load_and_preprocess_data(DATA_PATH)

    # Check if any valid data was loaded
    if not questions:
        print(" No questions loaded. Exiting preprocessing and training pipeline.")
        return

    # --- Step 2: Tokenize and Sequence Data ---
    print("\n--- Step 2: Tokenizing and Sequencing Data ---")
    tokenizer, vocab_size, encoder_input_data, decoder_input_data, decoder_target_data = \
        tokenize_and_sequence(questions, answers)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Encoder input shape: {encoder_input_data.shape}")
    print(f"Decoder input shape: {decoder_input_data.shape}")
    print(f"Decoder target shape: {decoder_target_data.shape}")

    # --- Step 3: Save Tokenizer and Processed Data ---
    print("\n--- Step 3: Saving Tokenizer and Processed Data ---")
    save_data(tokenizer, encoder_input_data, decoder_input_data, decoder_target_data)

    # --- Step 4: Train the Seq2Seq Model ---
    print("\n--- Step 4: Training the Seq2Seq Model ---")
    train_model(vocab_size, encoder_input_data, decoder_input_data, decoder_target_data)

    print("\n Chatbot training pipeline completed!")

if __name__ == '__main__':
    # Ensure 'data' directory exists before attempting to read/write files
    os.makedirs('data', exist_ok=True)
    main()