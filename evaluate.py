import tensorflow as tf
import process_data

encoder_input_data, decoder_input_data, decoder_target_data = process_data.process(is_train=False)

base_model = tf.keras.models.load_model('baseline_s2s.h5')
base_model.summary()

results = base_model.evaluate([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=32)
print('test loss, test acc:', results)
