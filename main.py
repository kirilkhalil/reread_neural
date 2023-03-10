import pandas as pd
import numpy as np
from tensorflow.keras import layers

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

talo_train = pd.read_csv('../talo_train_data.csv')
talo_np = np.array(talo_train)
print(talo_np)

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="character",
    output_mode="int",
)

vectorize_layer.adapt(talo_np)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
model.compile('rmsprop', 'mse')


test = model.predict(talo_train)
print(test.shape)