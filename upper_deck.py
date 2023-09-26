import numpy as np
import tensorflow as tf
import codecs as c
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
    file = c.open(filename, 'r', encoding='utf-16')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


raw_input_text = load_doc('finnish_upper_deck_inputs.txt')
input_lines = raw_input_text.split()
chars = sorted(list(set(raw_input_text)))  # All the separate chars found in input text
print(chars)
chars.remove(' ')
print(chars)
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
print(mapping)
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
print(input_hot.shape)

raw_labels = load_doc('finnish_upper_deck_labels.txt')
label_lines = raw_labels.split()
class_count = len(set(label_lines))
label_lines = np.array(label_lines)
output_hot = to_categorical(label_lines, class_count)
target_vector_length = len(output_hot[0])
print(output_hot.shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(word_length, vocab_size)))
model.add(tf.keras.layers.Dense(target_vector_length, activation='sigmoid'))
print(model.summary())
model.compile(loss=tf.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.9, momentum=0.2),
              metrics=['accuracy'],
              )
epochs = 1000
history = model.fit(input_hot, output_hot, epochs=epochs)
print("Evaluate model on test data")
results = model.evaluate(input_hot, output_hot, batch_size=128)
print("test loss, test acc:", results)
test_input = input_hot[0].reshape(1, word_length, vocab_size)
output = model(test_input)
output = np.array(output)
print(output)
print(output.shape)
print(np.argmax(output))
model.save('finnish_upper_deck.h5')
dump(mapping, open('finnish_upper_deck_mapping.pkl', 'wb'))
