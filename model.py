import tensorflow as tf

latent_dim = 40  # Latent dimensionality of the encoding space.
num_encoder_tokens = 11
num_decoder_tokens = 13
batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.

def seq2seq(is_compiled=True, num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens):
    # Define an input sequence and process it.
    encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    if is_compiled :
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return model