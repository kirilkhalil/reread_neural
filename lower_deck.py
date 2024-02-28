import numpy as np
import tensorflow as tf
import weight_multiplier
import output_evaluation
import codecs as c
from keras.utils import to_categorical
from pickle import dump
from strenum import StrEnum


tf.keras.utils.set_random_seed(
    24
)
np.set_printoptions(threshold=np.inf)


class FilePathEnums(StrEnum):
    FRCORPUS = 'french_corpus.txt'
    FRLDMODEL = 'french_lower_deck.h5'
    FRLDMAPPING = 'french_lower_deck_mapping.pkl'
    FRPOSSUPCORPUS = 'french_positional_supervised_corpus.txt'
    FRUDMODEL = 'french_upper_deck.h5'
    FRUDMAPPING = 'french_upper_deck_mapping.pkl'
    FRTWODECKTWORDS = 'french_two_deck_target_words.txt'
    FICORPUS = 'finnish_corpus.txt'
    FILDMODEL = 'finnish_lower_deck.h5'
    FILDMAPPING = 'finnish_lower_deck_mapping.pkl'
    FIPOSSUPCORPUS = 'finnish_positional_supervised_corpus.txt'
    FIUDMODEL = 'finnish_upper_deck.h5'
    FIUDMAPPING = 'finnish_upper_deck_mapping.pkl'
    FITWODECKTWORDS = 'finnish_two_deck_target_words.txt'


def load_doc(filename):
    # open the file as read only
    file = c.open(filename, 'r', encoding='utf-16')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def corpus_instantiation(language):  # Add cases as required for new language options. Make sure new entries follow the same ordering!
    setup_array = []
    if language == 'FIN':
        setup_array = [FilePathEnums.FICORPUS, FilePathEnums.FIPOSSUPCORPUS, FilePathEnums.FILDMODEL,
                       FilePathEnums.FIUDMODEL, FilePathEnums.FILDMAPPING, FilePathEnums.FIUDMAPPING, FilePathEnums.FITWODECKTWORDS]
    elif language == 'FR':
        setup_array = [FilePathEnums.FRCORPUS, FilePathEnums.FRPOSSUPCORPUS, FilePathEnums.FRLDMODEL,
                       FilePathEnums.FRUDMODEL, FilePathEnums.FRLDMAPPING, FilePathEnums.FRUDMAPPING, FilePathEnums.FRTWODECKTWORDS]
    else:
        print('No valid language chosen for corpus.')
    return setup_array


corpus_choices = ['FIN', 'FR']
chosen_corpus = corpus_choices[0]  # Choose language.
chosen_language = chosen_corpus
chosen_corpus = corpus_instantiation(chosen_corpus)
print(chosen_corpus[1])
raw_input_text = load_doc(chosen_corpus[1])
input_lines = raw_input_text.split()
print(len(input_lines))
chars = sorted(list(set(raw_input_text)))  # All the separate chars found in input text
chars.remove(' ')
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
print(mapping)
vocab_size = len(mapping)  # Size of vocabulary
print(vocab_size)
sequences = list()
word_length = 0
for line in input_lines:
    if len(line) > word_length:  # Figure out the longest input word length. Used also for padding length if needed.
        word_length = len(line)
    # print([mapping[char] for char in line])
    encoded_seq = [mapping[char] for char in line]
    sequences.append(encoded_seq)
sequences = np.array(sequences)
input_hot = to_categorical(sequences, vocab_size)
print(input_hot.shape)
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)

raw_output_text = load_doc(chosen_corpus[6])
output_lines = raw_output_text.split()
# print(output_lines)
output_sequences = list()
for line in output_lines:
    encoded_seq = [mapping[char] for char in line]
    output_sequences.append(encoded_seq)
output_sequences = np.array(output_sequences)
output_hot = to_categorical(output_sequences)
flattened_target = list()
for output_matrix in output_hot:
    flattened_target.append(output_matrix.flatten())
flattened_target = np.array(flattened_target)
# print(flattened_target.shape)

initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
position_count = 6  # How many positional chars are used in the inputs. In our standard case 6 x '#'.
last_layer_size = (word_length - position_count) * vocab_size
# print(last_layer_size)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(word_length, vocab_size)))
model.add(tf.keras.layers.Dense(118,
                                activation='sigmoid', kernel_initializer=initializer, name='hidden_layer'))
# input words rounded up to INT: sqrt(word_count * word_length) = node count
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(last_layer_size, activation='sigmoid', name='output_layer'))
print(model.summary())
model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=100, momentum=0.5),
              metrics=['mean_squared_error'],
              )
epochs = 10
history = model.fit(weighted_inputs, flattened_target, epochs=epochs)

layer_name = 'hidden_layer'
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                          outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(weighted_inputs)
print(intermediate_output[0])
int_output = np.array(intermediate_output[0])
chosen_output = int_output.argmax()
print(chosen_output)
print(int_output[chosen_output])

# for x in range(0, 13895):
#     test_input = weighted_inputs[x].reshape(1, word_length, vocab_size)
#     output = model(test_input)
#     output = np.array(output)
#     print(output_evaluation.output_eval(output))

model_name = ''
mapping_name = ''
if chosen_language == 'FIN':
    model_name = 'finnish_lower_deck.h5'
    mapping_name = 'finnish_lower_deck_mapping.pkl'
elif chosen_language == 'FR':
    model_name = 'french_lower_deck.h5'
    mapping_name = 'french_lower_deck_mapping.pkl'
else:
    chosen_language = ''
if chosen_language:
    model.save(model_name)
    dump(mapping, open(mapping_name, 'wb'))

