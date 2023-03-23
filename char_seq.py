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


# load
in_filename = '../talo_words.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
lines[0] = 'talo###'

chars = sorted(list(set(raw_text)))
#chars contains line change and \ufeff values so we remove those as we don't want to map those.
chars.remove('\n')
chars.remove('\ufeff')
mapping = dict((c, i) for i, c in enumerate(chars))

sequences = list()
for line in lines:
    # integer encode line
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)

vocab_size = len(mapping)

#Last element is empty due to using txt file and weird encoding stuff, so we pop it for now to make life easier.
sequences.pop()
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)


# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=110, verbose=2)

