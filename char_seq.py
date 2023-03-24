import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)


# Turns out this network is likely optimized to predict the next most likely word in a sequence.
# Need to re-think the network more in the lines of a Reconstruction LSTM autoencoder instead of a prediction one

# load input into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


in_filename = '../talo_words.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
lines[0] = 'talo###'

chars = sorted(list(set(raw_text)))
# chars contains line change and \ufeff values so we remove those as we don't want to map those.
chars.remove('\n')
chars.remove('\ufeff')
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)

vocab_size = len(mapping)

# Last element is empty due to using txt file and weird encoding stuff, so we pop it for now to make life easier.
sequences.pop()
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
              metrics=['accuracy'])
model.fit(X, y, epochs=110, verbose=2)
