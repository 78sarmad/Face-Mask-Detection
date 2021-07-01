import tensorflow as tf

model = tf.keras.models.load_model('pattern_model.h5')
tf.saved_model.save(model, 'pattern_detector')
