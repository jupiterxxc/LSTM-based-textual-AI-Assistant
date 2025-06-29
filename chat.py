import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import random
import time
import argparse

# --- Configuration ---
MAX_INPUT_LEN = 20  # Same as in preprocess_train.py
MAX_RESPONSE_LEN = 30  # Maximum response length
MODEL_PATH = 'chatbot_seq2seq_model.keras'
ENCODER_MODEL_PATH = 'encoder_model.keras'
DECODER_MODEL_PATH = 'decoder_model.keras'
TOKENIZER_PATH = 'data/tokenizer.pkl'


# --- Text cleaning function ---
def clean_text(text):
    """Clean and normalize input text."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# --- Load trained model and tokenizer ---
def load_resources(use_inference_models=True):
    print("📦 Loading tokenizer and model...")

    try:
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)

        if use_inference_models:
            # Load inference models if available
            try:
                encoder_model = tf.keras.models.load_model(ENCODER_MODEL_PATH)
                decoder_model = tf.keras.models.load_model(DECODER_MODEL_PATH)
                print("✅ Inference models loaded successfully.")
                return tokenizer, encoder_model, decoder_model
            except:
                print("⚠️ Inference models not found. Using full model instead.")
                model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Full model loaded successfully.")
                return tokenizer, model, None
        else:
            # Load full model
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Full model loaded successfully.")
            return tokenizer, model, None

    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        raise


# --- Preprocess user input for model ---
def preprocess_input(text, tokenizer):
    # Clean and normalize text
    text = clean_text(text)

    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence
    padded = pad_sequences(sequence, maxlen=MAX_INPUT_LEN, padding='post')

    return padded


# --- Beam search response generation ---
def beam_search_decode(input_seq, tokenizer, encoder_model, decoder_model, beam_width=3, max_length=MAX_RESPONSE_LEN):
    # Get initial states from encoder
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Get start token
    start_token = tokenizer.word_index.get('start', 1)
    end_token = tokenizer.word_index.get('end', 2)

    # Initialize with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    # Initialize beam search
    beams = [(target_seq, states_value, 0.0, [])]  # (sequence, states, score, words)
    completed_beams = []

    # Beam search loop
    for _ in range(max_length):
        all_candidates = []

        # Expand each beam
        for seq, states, score, words in beams:
            if words and words[-1] == 'end':
                completed_beams.append((seq, states, score, words))
                continue

            # Predict next tokens for this beam
            output_tokens, h, c = decoder_model.predict([seq] + states, verbose=0)
            states_value = [h, c]

            # Get top k predictions
            output_tokens = output_tokens[0, -1, :]
            top_indices = np.argsort(output_tokens)[-beam_width:]

            # Create new candidates
            for idx in top_indices:
                if idx == 0:  # Skip padding
                    continue

                word = tokenizer.index_word.get(idx, '')
                if not word:
                    continue

                # Create new sequence by appending token
                new_seq = np.zeros((1, seq.shape[1] + 1))
                new_seq[0, :-1] = seq[0]
                new_seq[0, -1] = idx

                # Calculate new score (log probability)
                new_score = score + np.log(output_tokens[idx])
                new_words = words + [word]

                # Add to candidates
                all_candidates.append((new_seq, states_value, new_score, new_words))

        # If no candidates, break
        if not all_candidates:
            break

        # Sort candidates by score and keep top beam_width
        all_candidates.sort(key=lambda x: -x[2])  # Sort by score (descending)
        beams = all_candidates[:beam_width]

        # If all beams are completed, break
        if len(completed_beams) == beam_width:
            break

    # Combine completed beams and current beams
    all_beams = completed_beams + beams
    all_beams.sort(key=lambda x: -x[2])  # Sort by score

    # Get best beam's words
    best_words = all_beams[0][3]

    # Filter out start and end tokens and clean up
    response_words = [word for word in best_words if word not in ['start', 'end', '<OOV>']]

    # Clean up response
    response = ' '.join(response_words)
    response = clean_response(response)

    return response


# --- Temperature-based sampling for more varied responses ---
def generate_with_sampling(input_seq, tokenizer, model, temperature=0.8, max_length=MAX_RESPONSE_LEN):
    start_token = tokenizer.word_index.get('start', 1)
    end_token = tokenizer.word_index.get('end', 2)

    # Initialize with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = start_token

    decoded_sentence = []
    stop_condition = False

    while not stop_condition:
        # Predict next token
        output_tokens = model.predict([input_seq, target_seq], verbose=0)

        # Sample from the output distribution
        sampled_token_index = sample_with_temperature(output_tokens[0, -1, :], temperature)

        # Get the word for this token
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        # Check for end conditions
        if (sampled_word == 'end' or sampled_word == '' or
                len(decoded_sentence) >= max_length):
            stop_condition = True
        else:
            # Don't add special tokens to the output
            if sampled_word not in ['start', '<OOV>']:
                decoded_sentence.append(sampled_word)

            # Update the target sequence
            target_seq = np.concatenate(
                [target_seq, np.array([[sampled_token_index]])],
                axis=-1
            )

            # Check for repetition
            if len(decoded_sentence) >= 3:
                # Stop if we're repeating the same word
                if len(set(decoded_sentence[-3:])) == 1:
                    stop_condition = True

    # Clean up response
    response = ' '.join(decoded_sentence)
    response = clean_response(response)

    return response


# --- Temperature sampling helper ---
def sample_with_temperature(probabilities, temperature=0.8):
    """Sample from probability distribution with temperature."""
    # Apply temperature to soften/sharpen distribution
    probabilities = np.asarray(probabilities).astype('float64')
    probabilities = np.log(np.maximum(probabilities, 1e-10)) / temperature
    exp_probs = np.exp(probabilities - np.max(probabilities))
    probabilities = exp_probs / np.sum(exp_probs)

    # Sometimes use argmax, sometimes sample
    if random.random() < 0.3:  # 30% of the time, use argmax
        return np.argmax(probabilities)
    else:  # 70% of the time, sample
        return np.random.choice(len(probabilities), p=probabilities)


# --- Clean up response ---
def clean_response(response):
    """Clean up the generated response."""
    # Remove repetitions of 3+ words
    words = response.split()
    if len(words) <= 3:
        return response

    # Filter out repetitive patterns
    cleaned_words = []
    i = 0
    while i < len(words):
        cleaned_words.append(words[i])

        # Check for repetition patterns
        if i >= 2 and words[i] == words[i - 1] == words[i - 2]:
            # Skip until we find a different word
            while i + 1 < len(words) and words[i + 1] == words[i]:
                i += 1

        i += 1

    # Join and limit length
    cleaned = ' '.join(cleaned_words)

    # Cap response length
    if len(cleaned_words) > 30:
        cleaned = ' '.join(cleaned_words[:30])

    # Capitalize first letter and add period if missing
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'

    return cleaned


# --- Generate chatbot response ---
def generate_response(input_text, tokenizer, model, decoder_model=None, use_beam_search=True):
    # Preprocess input
    input_seq = preprocess_input(input_text, tokenizer)

    try:
        # If we have inference models, use beam search
        if decoder_model is not None and use_beam_search:
            response = beam_search_decode(input_seq, tokenizer, model, decoder_model)
        else:
            # Otherwise use temperature sampling with the full model
            response = generate_with_sampling(input_seq, tokenizer, model)

        # Fallback for empty responses
        if not response.strip():
            fallbacks = [
                "I'm not sure how to respond to that.",
                "Could you elaborate more?",
                "Interesting point. Could you tell me more?",
                "I'm still learning about that topic.",
                "That's something I'd like to learn more about."
            ]
            response = random.choice(fallbacks)

    except Exception as e:
        print(f"Error generating response: {e}")
        response = "I'm having trouble processing that right now."

    return response


# --- Interactive Chat loop ---
def chat(use_beam_search=True, use_inference_models=True, show_thinking=False):
    try:
        if use_inference_models:
            tokenizer, encoder_model, decoder_model = load_resources()
        else:
            tokenizer, model, _ = load_resources(use_inference_models=False)
            encoder_model, decoder_model = None, None
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    print("\n🤖 Chatbot is ready to chat! Type 'exit' to stop chatting.\n")
    print("💡 Try asking questions or making statements to see how the chatbot responds.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower().strip() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! 👋")
            break

        start_time = time.time()

        if show_thinking:
            print("Chatbot: 🤔 Thinking...")

        if use_inference_models:
            response = generate_response(user_input, tokenizer, encoder_model, decoder_model, use_beam_search)
        else:
            response = generate_response(user_input, tokenizer, model)

        end_time = time.time()
        process_time = end_time - start_time

        print(f"Chatbot: {response}")
        if show_thinking:
            print(f"(Response generated in {process_time:.2f} seconds)")


# --- Command line arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='Interactive chatbot')
    parser.add_argument('--no-beam', action='store_true', help='Disable beam search (use sampling instead)')
    parser.add_argument('--full-model', action='store_true', help='Use full model instead of separate encoder/decoder')
    parser.add_argument('--show-time', action='store_true', help='Show response generation time')
    return parser.parse_args()


# --- Run chatbot ---
if __name__ == '__main__':
    args = parse_args()
    chat(
        use_beam_search=not args.no_beam,
        use_inference_models=not args.full_model,
        show_thinking=args.show_time
    )