import pandas as pd
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras import layers

np.set_printoptions(threshold=np.inf)

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

talo_train = pd.read_csv('../talo_train_data.csv')
talo_np = np.array(talo_train)
#print(talo_np)

vectorize_layer = layers.TextVectorization(
    standardize=None,
    split="character",
    output_mode="int",
)

#vectorize_layer.adapt(talo_np)

# talo_np = vectorize_layer(talo_np)
#print(talo_np)


# one_hot = to_categorical(talo_np, num_classes=7)
#print(one_hot)


# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(None,), dtype="int64"))


# test = model.predict(talo_train)

