import numpy as np
import tensorflow as tf
import weight_multiplier
from tensorflow.keras import layers
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
chars.remove('\n')
chars.remove(' ')
# print(chars)
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
# print(mapping)
sequences = list()
for line in input_lines:
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)
vocab_size = len(mapping)  # Size of vocabulary
# print(vocab_size)
sequences = np.array(sequences)
# print(sequences)


raw_output_text = load_doc('../word_list.rtf')
output_lines = raw_output_text.split()


vectorize_layer = layers.TextVectorization(
    standardize=None,
    split="character",
    output_mode="int",
)

vectorize_layer2 = layers.TextVectorization(
    standardize=None,
    output_mode="int",
)

vectorize_layer.adapt(input_lines)
# print(lines)
vec_text = vectorize_layer(input_lines)
# print(vec_text)
lines_array = np.array(vec_text)
lines_array[lines_array == 2] = -1
# print(lines_array[0])
input_hot = to_categorical(sequences, vocab_size)
# print(input_hot[0])
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)

vectorize_layer2.adapt(input_lines)
target_vec = vectorize_layer2(input_lines)
t_lines_array = np.array(target_vec)
# print(t_lines_array)
target_hot = to_categorical(t_lines_array)
# print(target_hot.shape)


# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(13, 29)))
# model.add(tf.keras.layers.Dense(60,
#                                 activation='sigmoid'))  # Number of neurons with 500 input words rounded up to INT:
# # sqrt(500 * 7) = 60
# model.add(tf.keras.layers.Dense(3502, activation='sigmoid'))
#
# print(model.summary())
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
#               metrics=['accuracy'],
#               )
# epochs = 30
# history = model.fit(input_hot, target_hot, epochs=epochs)
#
# print("Evaluate model on test data")
# results = model.evaluate(input_hot, target_hot, batch_size=128)
# print("test loss, test acc:", results)
#
# test_input = input_hot[0].reshape(1, 13, 29)
# output = model(test_input)
# output = np.array(output)
# print(output)
# print(np.argmax(output))
