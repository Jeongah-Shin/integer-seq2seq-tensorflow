import process_data
import tensorflow as tf

encoder_input_data, decoder_input_data, decoder_target_data = process_data.process(is_train=False)
base_model = tf.keras.models.load_model('./pretrained_weights/t1_baseline.h5')

print(str(encoder_input_data.shape))
print(str(decoder_input_data.shape))
print(str(decoder_target_data.shape))

for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoder_input = decoder_input_data[seq_index: seq_index + 1]
    target_data = decoder_target_data[seq_index: seq_index + 1]

    decoded_sentence = base_model.predict([input_seq, decoder_input])

    print('input shape : ' + str(input_seq.shape))
    print(str(input_seq))
    print('output shape : ' + str(decoded_sentence.shape))
    print(str(decoded_sentence))
    print('result : ' + str(target_data == decoded_sentence))

    # print('This is decoded result : ' + str(decoded_sentence))