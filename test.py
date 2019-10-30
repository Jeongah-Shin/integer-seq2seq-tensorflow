import process_data
import numpy as np
from model import *

encoder_input_data, decoder_input_data, decoder_target_data = process_data.process()
base_model = tf.keras.models.load_model('baseline_s2s.h5')

results = base_model.evalueate([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32)
print('test loss, test acc:', results)

for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoder_input = decoder_input_data[seq_index: seq_index + 1]
    target_data = decoder_target_data[seq_index: seq_index + 1]
    # print(input_seq.shape)
    # decoded_sentence = base_model.predict([input_seq, decoder_input])
    # print('shape : ' + str(decoded_sentence.shape))

    # print('This is decoded result : ' + str(decoded_sentence))