import model as m
import process_data

encoder_input_data, decoder_input_data, decoder_target_data = process_data.process()
model = m.seq2seq()

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=m.batch_size,
          epochs=m.epochs,
          validation_split=0.2)
# Save model
model.save('baseline_s2s.h5')