import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load text
test_data = ["talo###", "#talo##", '##talo#', '###talo', 'lota###', '#lota##', '##lota#', '###lota'], ["talo###",
                                                                                                       "#talo##",
                                                                                                       '##talo#',
                                                                                                       '###talo',
                                                                                                       'lota###',
                                                                                                       '#lota##',
                                                                                                       '##lota#',
                                                                                                       '###lota']

target_data = ["talo###", "#talo##", '##talo#', '###talo', 'lota###', '#lota##', '##lota#', '###lota'], ["talo###",
                                                                                                       "#talo##",
                                                                                                       '##talo#',
                                                                                                       '###talo',
                                                                                                       'lota###',
                                                                                                       '#lota##',
                                                                                                       '##lota#',
                                                                                                       '###lota']

data = []
t_data = []
# Loop to create an array of Tensors to feed to NN:
for texts in test_data:
    for text in texts:
        data.append(np.fromstring(text, dtype=np.uint8) - ord('a'))

for texts2 in target_data:
    for text2 in texts2:
        t_data.append(np.fromstring(text2, dtype=np.uint8) - ord('a'))

# Need to change to handle an array of Tensors
one_hot_encode = tf.one_hot(data, 26, dtype=tf.uint8)
one_hot_encode2 = tf.one_hot(t_data, 26, dtype=tf.uint8)
#print(one_hot_encode.shape)


model = Sequential()
model.add(LSTM(75, input_shape=(one_hot_encode.shape[1], one_hot_encode.shape[2])))
model.add(Dense(7, activation="softmax"))(26)
#print(model.summary())


# compile model
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['accuracy'])
# fit model
print(one_hot_encode2.shape)
print(one_hot_encode.shape)
model.fit(one_hot_encode, one_hot_encode2, epochs=100, verbose=2)
