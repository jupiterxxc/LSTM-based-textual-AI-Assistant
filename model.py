import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Attention, Concatenate, TimeDistributed
from tensorflow.keras.models import Model

def create_seq2seq_chatbot_model(vocab_size, embedding_dim, lstm_units, max_input_len, max_target_len):

    # --- Encoder ---
    encoder_inputs = Input(shape=(max_input_len,), name='encoder_inputs')

    # Embedding layer for the encoder input
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)

    # Encoder LSTM layer:
    encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True, return_sequences=True, name='encoder_lstm')(encoder_embedding)

    # Store the encoder's final states, which will be the initial states for the decoder.
    encoder_states = [state_h, state_c]

    # --- Decoder ---
    decoder_inputs = Input(shape=(max_target_len,), name='decoder_inputs')

    # Embedding layer for the decoder input
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)

    # Decoder LSTM layer:
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # --- Attention Mechanism ---
    attention_layer = Attention(name='attention_layer')
    attention_output = attention_layer([decoder_outputs, encoder_outputs])

    # Concatenate the attention output with the decoder's LSTM output.
    decoder_concat_input = Concatenate(axis=-1, name='concat_attention_decoder')([decoder_outputs, attention_output])

    # TimeDistributed Dense layer for outputting probabilities over the vocabulary for each timestep.
    decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax', name='decoder_output_dense'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model:
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model