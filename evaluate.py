import tensorflow as tf
import process_data

encoder_input_data, decoder_input_data, decoder_target_data = process_data.process()

model = tf.keras.models.load_model('baseline_s2s.h5')
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = process_data.input_characters[seq_index: seq_index + 1]
    output_seq = process_data.target_characters[seq_index: seq_index + 1]
    loss, acc = model.evaluate(input_seq, output_seq, verbose=2)
    print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
