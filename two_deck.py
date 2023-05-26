import numpy as np
import tensorflow as tf
import weight_multiplier
import output_evaluation
from keras.utils import to_categorical
from pickle import dump


tf.keras.utils.set_random_seed(
    24
)
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
print(input_lines)

chars = sorted(list(set(raw_input_text)))  # All the separate chars found in input text
chars.remove(' ')
chars.remove('\n')
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
vocab_size = len(mapping)  # Size of vocabulary

sequences = list()
word_length = 0
for line in input_lines:
    if len(line) > word_length:  # Figure out the longest input word length. Used also for padding length if needed.
        word_length = len(line)
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)
sequences = np.array(sequences)
input_hot = to_categorical(sequences, vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)

raw_output_text = load_doc('two_deck_target_words.rtf')
output_lines = raw_output_text.split()
print(output_lines)
output_sequences = list()
chars.remove('#')  # Don't want '#' char as one of our output alphabet
for line in output_lines:
    encoded_seq = [mapping[char] for char in line]
    output_sequences.append(encoded_seq)
output_sequences = np.array(output_sequences)
print(output_sequences)
output_hot = to_categorical(output_sequences)
print(output_hot.shape)
flattened_target = list()
for output_matrix in output_hot:
    flattened_target.append(output_matrix.flatten())
print(flattened_target[0])
flattened_target = np.array(flattened_target)
# print(flattened_target)
print(flattened_target.shape)

initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(word_length, vocab_size)))
model.add(tf.keras.layers.Dense(120,
                                activation='sigmoid', kernel_initializer=initializer, ))  # Number of neurons with 500
# input words rounded up to INT: sqrt(500 * 7) = 60
model.add(tf.keras.layers.Dense(189, activation='sigmoid'))
print(model.summary())
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=200, momentum=0.2),
              metrics=['mean_squared_error'],
              )
epochs = 800
history = model.fit(weighted_inputs, flattened_target, epochs=epochs)
# print("Evaluate model on test data")
# results = model.evaluate(weighted_inputs, flattened_target, batch_size=128)
# print("test loss, test acc:", results)
for x in range(0, 49):
    test_input = weighted_inputs[x].reshape(1, word_length, vocab_size)
    output = model(test_input)
    output = np.array(output)
    output_evaluation.output_eval(output)
# print(output)
# print(output.shape)
# print(np.argmax(output))
# output_evaluation.output_eval(output)
model.save('lower_deck.h5')
dump(mapping, open('lower_deck_mapping.pkl', 'wb'))





