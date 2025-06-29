import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time
import difflib
from sklearn.metrics import classification_report, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MAX_INPUT_LEN = 20
MAX_TARGET_LEN = 20
MODEL_PATH = 'chatbot_seq2seq_model.keras'
TOKENIZER_PATH = 'data/tokenizer.pkl'

# --- Load tokenizer and model ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError("Model or tokenizer file not found! Please ensure 'chatbot_seq2seq_model.keras' and 'data/tokenizer.pkl' exist.")

with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model(MODEL_PATH)
print(" Model and tokenizer loaded successfully!\n")

# --- Sample test data ---
test_samples = [
    {"input": "Hi", "expected": "hello"},
    {"input": "How are you?", "expected": "i'm good how about you"},
    {"input": "What is AI?", "expected": "ai stands for artificial intelligence"},
    {"input": "Tell me a joke", "expected": "why dont scientists trust atoms because they make up everything"},
    {"input": "Thank you", "expected": "you are welcome"},
    {"input": "What is the capital of France?", "expected": "paris is the capital of france"},
    {"input": "Do you like music?", "expected": "yes i enjoy music"},
    {"input":"Bye", "expected": "goodbye have a nice day"},
    {"input":"Can you dance","expected":"I wish I could but I can only chat"},
    {"input":"Who are you","expected":"I am a chatbot a conversational ai."},
    {"input":"Is there any chocolate cake available","expected":"Yes Chocolate cakes are available."},
    {"input":"Can i order a chocolate cake","expected":"Yes we make custom cakes just the way you like."},
    {"input":"Where can i order the Cake","expected":"You can order online by phone or in the shop."},
    {"input":"How do i place an order","expected":"you can order directly through our website or you can give us a call if you need assistance"}
]

# --- Preprocess input ---
def preprocess_input(text):
    sequence = tokenizer.texts_to_sequences([text.lower().strip()])
    padded = pad_sequences(sequence, maxlen=MAX_INPUT_LEN, padding='post')
    return padded

# --- Generate response ---
def generate_response(input_text):
    input_seq = preprocess_input(input_text)
    # Ensure 'start' and 'end' tokens are in tokenizer, default to 1 and 2 if not found
    # (assuming 0 is padding, 1 is 'start', 2 is 'end' based on common practice)
    start_token_index = tokenizer.word_index.get('start', 1)
    end_token_index = tokenizer.word_index.get('end', 2)

    decoder_input = np.array([[start_token_index]]) # Start with 'start' token
    decoded_sentence = []

    for _ in range(MAX_TARGET_LEN):
        # Predict the next token
        # verbose=0 suppresses the prediction progress bar for cleaner output
        output_tokens = model.predict([input_seq, decoder_input], verbose=0)
        # Get the token with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # If it's the end token or padding token (0), stop generation
        if sampled_token_index in [end_token_index, 0]:
            break

        # Convert token index back to word
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')
        if sampled_word: # Ensure word is not empty
            decoded_sentence.append(sampled_word)

        # Append the sampled token to the decoder input for the next step
        decoder_input = np.append(decoder_input, [[sampled_token_index]], axis=-1)

    return ' '.join(decoded_sentence)

# --- Evaluate model ---
def evaluate_model(samples):
    correct = 0
    y_true_raw, y_pred_raw = [], [] # Store raw sentences for BLEU and report
    total_bleu = 0
    smoother = SmoothingFunction().method4 # Smoothing for BLEU score

    start_time = time.time()

    print(" Running test cases...\n")
    for idx, sample in enumerate(samples, 1):
        input_text = sample['input']
        expected = sample['expected'].lower()
        predicted = generate_response(input_text).lower()

        # Calculate similarity for "correct" check
        similarity = difflib.SequenceMatcher(None, expected, predicted).ratio()
        # Define correctness: exact match or high string similarity
        is_correct = expected in predicted or similarity > 0.8

        print(f"🔹 Test {idx}")
        print(f"Input     : {input_text}")
        print(f"Expected  : {expected}")
        print(f"Predicted : {predicted}")
        print(f"Result    : {' Pass' if is_correct else ' Fail'}\n")

        if is_correct:
            correct += 1

        # BLEU Score calculation
        # sentence_bleu expects a list of reference sentences (even if just one)
        bleu_score = sentence_bleu([expected.split()], predicted.split(), smoothing_function=smoother)
        total_bleu += bleu_score

        y_true_raw.append(expected)
        y_pred_raw.append(predicted)

    total = len(samples)
    accuracy = (correct / total) * 100
    avg_bleu = total_bleu / total
    elapsed = time.time() - start_time

    print(" Evaluation Summary:")
    print(f" Accuracy     : {accuracy:.2f}%")
    print(f" BLEU Score   : {avg_bleu:.2f}")
    print(f" Total Samples: {total}")
    print(f" Time Taken   : {elapsed:.2f} sec")

    # --- Classification Report (simplified for string labels) ---
    print("\n Classification Report (simplified):")
    # Note: Classification report works best when mapping strings to integers if
    # there are many unique labels. For direct strings, it's very sparse.
    # We use y_true_raw and y_pred_raw directly as strings here.
    print(classification_report(y_true_raw, y_pred_raw, zero_division=0))

    # --- Confusion Matrix (Improved Visualization with original words) ---
    print("\n Confusion Matrix:")
    # Get all unique labels from both true and predicted values
    all_labels_set = sorted(list(set(y_true_raw + y_pred_raw)))

    # Create a mapping from string label to numerical index for calculation
    # (still useful for internal calculation, but not for direct plotting ticks)
    label_to_index = {label: i for i, label in enumerate(all_labels_set)}

    # Convert true and predicted string labels to numerical indices for `confusion_matrix` function
    y_true_indexed = [label_to_index[label] for label in y_true_raw]
    y_pred_indexed = [label_to_index[label] for label in y_pred_raw]

    # Calculate the confusion matrix using numerical indices
    cm = confusion_matrix(y_true_indexed, y_pred_indexed, labels=np.arange(len(all_labels_set)))
    print(cm)

    # --- Plot Confusion Matrix ---
    # Significantly increased figure size for better readability of long labels
    plt.figure(figsize=(20, 18)) # Increased size even further
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=18) # Increased title font size
    plt.colorbar()

    tick_marks = np.arange(len(all_labels_set))
    # Use the actual sorted labels for ticks, rotate them vertically
    plt.xticks(tick_marks, all_labels_set, rotation=90, ha='right', fontsize=8) # Words on X-axis, rotated
    plt.yticks(tick_marks, all_labels_set, fontsize=8) # Words on Y-axis

    plt.xlabel('Predicted Response', fontsize=14) # Clearer label
    plt.ylabel('True Response', fontsize=14) # Clearer label

    # Add text annotations for cell values
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Only display values if cell is not 0 for clarity
            if cm[i, j] > 0:
                plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black', fontsize=10)

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

    # The mapping from index to label is no longer strictly necessary to print
    # as the labels are directly on the plot now, but it can be kept for reference.
    # print("\n--- Confusion Matrix Index Mapping (for reference) ---")
    # for index, label in index_to_label.items():
    #     print(f"Index {index}: {label}")
    # print("------------------------------------------------------")


# --- Main entry ---
if __name__ == "__main__":
    evaluate_model(test_samples)
