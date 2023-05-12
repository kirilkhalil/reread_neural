import numpy as np
import tensorflow as tf
import weight_multiplier
import output_evaluation
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
vocab_size = len(mapping)  # Size of vocabulary

sequences = list()
for line in input_lines:
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
model.add(tf.keras.layers.Flatten(input_shape=(13, 27)))
model.add(tf.keras.layers.Dense(60,
                                activation='sigmoid', kernel_initializer=initializer))  # Number of neurons with 500
# input words rounded up to INT: sqrt(500 * 7) = 60
model.add(tf.keras.layers.Dense(189, activation='sigmoid'))
print(model.summary())
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.9, momentum=0.2),
              metrics=['mean_squared_error'],
              )
epochs = 500
history = model.fit(weighted_inputs, flattened_target, epochs=epochs)
# print("Evaluate model on test data")
# results = model.evaluate(weighted_inputs, flattened_target, batch_size=128)
# print("test loss, test acc:", results)
test_input = weighted_inputs[0].reshape(1, 13, 27)  # ######ABILITY
output = model(test_input)  # Expected output is: ABILITY
output = np.array(output)
# print(output)
# print(output.shape)
# print(np.argmax(output))
output_evaluation.output_eval(output)



