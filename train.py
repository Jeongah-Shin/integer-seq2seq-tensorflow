import tensorflow as tf
import model as m
import process_data
import matplotlib.pyplot as plt

num_samples = 7260  # Number of samples to train on.

encoder_input_data, decoder_input_data, decoder_target_data = process_data.process()
model = m.seq2seq()

callbacks = [
# If 'val_loss' does not improve over 2 epochs, the training stops.
tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
# Record logs for displaying on tensor board
tf.keras.callbacks.TensorBoard(log_dir='./tensor_board')
]

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    callbacks=callbacks,
                    batch_size=m.batch_size, epochs=m.epochs, validation_split=0.2)

# Save model
model.save_weights('./pretrained_weights/t1_savedModel', save_format='tf')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('3_training_loss.png')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('2_training_acc.png')
plt.show()