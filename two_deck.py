import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


raw_input_text = load_doc('../positional_corpus.rtf')
input_lines = raw_input_text.split()
chars = sorted(list(set(raw_input_text)))  # All the separate chars found in input text
chars.remove(' ')
chars.remove('\n')
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
sequences = list()
for line in input_lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)
vocab_size = len(mapping)  # Size of vocabulary
sequences = np.array(sequences)
input_hot = to_categorical(sequences, vocab_size)
for input_matrix in input_hot:  # We don't want the char '#' to have an active bit (1) in its representation
    for input_vector in input_matrix:
        if input_vector[0] == 1:
            input_vector[0] = 0

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(13, 27)))
# model.add(tf.keras.layers.Dense(60,
#                                 activation='sigmoid'))  # Number of neurons with 500 input words rounded up to INT:
# # sqrt(500 * 7) = 60
# model.add(tf.keras.layers.Dense(7, 27, activation='sigmoid'))
# print(model.summary())
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
#               metrics=['accuracy'],
#               )
# epochs = 40
# history = model.fit(weighted_inputs, output_hot, epochs=epochs)
