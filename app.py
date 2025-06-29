import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import random
import time
import argparse
from flask import Flask, request, render_template

# --- Configuration ---
MAX_INPUT_LEN = 20
MAX_RESPONSE_LEN = 30
MODEL_PATH = 'chatbot_seq2seq_model.keras'
ENCODER_MODEL_PATH = 'encoder_model.keras'
DECODER_MODEL_PATH = 'decoder_model.keras'
TOKENIZER_PATH = 'data/tokenizer.pkl'

app = Flask(__name__)

# --- Global variables to hold loaded resources ---
tokenizer = None
encoder_model = None
decoder_model = None
full_model = None
use_beam_search_global = True
use_inference_models_global = False

# --- Text cleaning function ---
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# --- Load trained model and tokenizer ---
def load_resources(use_inference_models):
    print(" Loading tokenizer and model...")
    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        print(f" Error loading tokenizer: {e}")
        raise

    if use_inference_models:
        try:
            encoder_model = tf.keras.models.load_model(ENCODER_MODEL_PATH)
            decoder_model = tf.keras.models.load_model(DECODER_MODEL_PATH)
            print(" Inference models loaded successfully.")
            return tokenizer, encoder_model, decoder_model, None
        except Exception as e:
            print(f" Inference models not found. Using full model instead. Error: {e}")
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                print(" Full model loaded successfully.")
                return tokenizer, None, None, model
            except Exception as e:
                print(f" Error loading full model: {e}")
                raise
    else:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(" Full model loaded successfully.")
            return tokenizer, None, None, model
        except Exception as e:
            print(f" Error loading full model: {e}")
            raise

# --- Preprocess user input for model ---
def preprocess_input(text, tokenizer):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_INPUT_LEN, padding='post')
    return padded

# --- Beam search response generation ---
def beam_search_decode(input_seq, tokenizer, encoder_model, decoder_model, beam_width=3, max_length=MAX_RESPONSE_LEN):
    states_value = encoder_model.predict(input_seq, verbose=0)
    start_token = tokenizer.word_index.get('start', 1)
    end_token = tokenizer.word_index.get('end', 2)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    beams = [(target_seq, states_value, 0.0, [])]
    completed_beams = []

    for _ in range(max_length):
        all_candidates = []
        for seq, states, score, words in beams:
            if words and words[-1] == 'end':
                completed_beams.append((seq, states, score, words))
                continue

            output_tokens, h, c = decoder_model.predict([seq] + states, verbose=0)
            states_value = [h, c]
            output_tokens = output_tokens[0, -1, :]
            top_indices = np.argsort(output_tokens)[-beam_width:]

            for idx in top_indices:
                if idx == 0:
                    continue
                word = tokenizer.index_word.get(idx, '')
                if not word:
                    continue
                new_seq = np.zeros((1, seq.shape[1] + 1))
                new_seq[0, :-1] = seq[0]
                new_seq[0, -1] = idx
                new_score = score + np.log(output_tokens[idx])
                new_words = words + [word]
                all_candidates.append((new_seq, states_value, new_score, new_words))

        if not all_candidates:
            break
        all_candidates.sort(key=lambda x: -x[2])
        beams = all_candidates[:beam_width]
        if len(completed_beams) == beam_width:
            break

    all_beams = completed_beams + beams
    all_beams.sort(key=lambda x: -x[2])
    best_words = all_beams[0][3]
    response_words = [word for word in best_words if word not in ['start', 'end', '<OOV>']]
    response = ' '.join(response_words)
    response = clean_response(response)
    return response

# --- Temperature-based sampling ---
def generate_with_sampling(input_seq, tokenizer, model, temperature=0.8, max_length=MAX_RESPONSE_LEN):
    start_token = tokenizer.word_index.get('start', 1)
    end_token = tokenizer.word_index.get('end', 2)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    decoded_sentence = []
    stop_condition = False

    while not stop_condition:
        output_tokens = model.predict([input_seq, target_seq], verbose=0)
        sampled_token_index = sample_with_temperature(output_tokens[0, -1, :], temperature)
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if (sampled_word == 'end' or sampled_word == '' or len(decoded_sentence) >= max_length):
            stop_condition = True
        else:
            if sampled_word not in ['start', '<OOV>']:
                decoded_sentence.append(sampled_word)

            target_seq = np.concatenate([target_seq, np.array([[sampled_token_index]])], axis=-1)

            if len(decoded_sentence) >= 3 and len(set(decoded_sentence[-3:])) == 1:
                stop_condition = True

    response = ' '.join(decoded_sentence)
    response = clean_response(response)
    return response

# --- Temperature sampling helper ---
def sample_with_temperature(probabilities, temperature=0.8):
    probabilities = np.asarray(probabilities).astype('float64')
    probabilities = np.log(np.maximum(probabilities, 1e-10)) / temperature
    exp_probs = np.exp(probabilities - np.max(probabilities))
    probabilities = exp_probs / np.sum(exp_probs)
    if random.random() < 0.3:
        return np.argmax(probabilities)
    else:
        return np.random.choice(len(probabilities), p=probabilities)

# --- Clean up response ---
def clean_response(response):
    words = response.split()
    if len(words) <= 3:
        return response
    cleaned_words = []
    i = 0
    while i < len(words):
        cleaned_words.append(words[i])
        if i >= 2 and words[i] == words[i - 1] == words[i - 2]:
            while i + 1 < len(words) and words[i + 1] == words[i]:
                i += 1
        i += 1
    cleaned = ' '.join(cleaned_words)
    if len(cleaned_words) > 30:
        cleaned = ' '.join(cleaned_words[:30])
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
    return cleaned

# --- Generate chatbot response ---
def generate_response(input_text):
    global tokenizer, encoder_model, decoder_model, full_model, use_beam_search_global, use_inference_models_global

    if tokenizer is None or full_model is None and (encoder_model is None or decoder_model is None):
        return "Error: Chatbot models not loaded correctly."

    input_seq = preprocess_input(input_text, tokenizer)
    try:
        if decoder_model is not None and use_beam_search_global:
            response = beam_search_decode(input_seq, tokenizer, encoder_model, decoder_model)
        elif full_model is not None:
            response = generate_with_sampling(input_seq, tokenizer, full_model)
        else:
            return "Error: Chatbot models not loaded correctly."
        return response
    except Exception as e:
        print(f"Error generating response from model: {e}")
        return "I'm having trouble processing that right now."

# --- Flask Routes ---
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = generate_response(user_input)
    return {"response": response}

# --- Command line arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='Interactive chatbot web app')
    parser.add_argument('--no-beam', action='store_true', help='Disable beam search (use sampling instead)')
    parser.add_argument('--full-model', action='store_true', help='Use full model instead of separate encoder/decoder')
    parser.add_argument('--host', default='127.0.0.1', help='Host for the web app')
    parser.add_argument('--port', type=int, default=5000, help='Port for the web app')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    use_beam_search_global = not args.no_beam
    use_inference_models_global = not args.full_model
    try:
        tokenizer, encoder_model, decoder_model, full_model = load_resources(use_inference_models=not args.full_model)
    except Exception as e:
        print(f"Failed to load resources: {e}")
        exit()
    app.run(host=args.host, port=args.port, debug=True)
