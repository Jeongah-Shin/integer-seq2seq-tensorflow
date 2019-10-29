import tensorflow as tf
import numpy as np
import model as m
import process_data

model = m.seq2seq()
encoder_input_data, decoder_input_data, decoder_target_data = process_data.process()

# Define sampling models
encoder_model = tf.keras.Model(model.encoder_inputs, model.encoder_states)

decoder_state_input_h = tf.keras.Input(shape=(model.latent_dim,))
decoder_state_input_c = tf.keras.Input(shape=(model.latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = model.decoder_lstm(
    model.decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = model.decoder_dense(decoder_outputs)
decoder_model = tf.keras.Model(
    [model.decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in process_data.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in process_data.target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, process_data.num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, process_data.target_token_index['s']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'e' or
           len(decoded_sentence) > process_data.max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, process_data.num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', process_data.input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)