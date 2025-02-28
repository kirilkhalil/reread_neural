import numpy as np
import tensorflow as tf
import codecs as c
import weight_multiplier
import output_evaluation
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
    FRUDECKLABELS = 'french_upper_deck_labels.txt'
    FICORPUS = 'finnish_corpus.txt'
    FILDMODEL = 'finnish_lower_deck.h5'
    FILDMAPPING = 'finnish_lower_deck_mapping.pkl'
    FIPOSSUPCORPUS = 'finnish_positional_supervised_corpus.txt'
    FIUDMODEL = 'finnish_upper_deck.h5'
    FIUDMAPPING = 'finnish_upper_deck_mapping.pkl'
    FITWODECKTWORDS = 'finnish_two_deck_target_words.txt'
    FIUDECKLABELS = 'finnish_upper_deck_labels.txt'
    FIRNDCORPUS = 'fin_random_corpus.txt'
    FIRNDLDMODEL = 'fin_random_lower_deck.h5'
    FIRNDLDMAPPING = 'fin_random_lower_deck_mapping.pkl'
    FIRNDPOSSUPCORPUS = 'fin_random_positional_supervised_corpus.txt'
    FIRNDUDMODEL = 'fin_random_upper_deck.h5'
    FIRNDUDMAPPING = 'fin_random_upper_deck_mapping.pkl'
    FIRNDTWODECKTWORDS = 'fin_random_two_deck_target_words.txt'
    FIRNDUDELABELS = 'finnish_random_upper_deck_labels.txt'


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
                       FilePathEnums.FIUDMODEL, FilePathEnums.FILDMAPPING, FilePathEnums.FIUDMAPPING, FilePathEnums.FITWODECKTWORDS, FilePathEnums.FRUDECKLABELS]
    elif language == 'FR':
        setup_array = [FilePathEnums.FRCORPUS, FilePathEnums.FRPOSSUPCORPUS, FilePathEnums.FRLDMODEL,
                       FilePathEnums.FRUDMODEL, FilePathEnums.FRLDMAPPING, FilePathEnums.FRUDMAPPING, FilePathEnums.FRTWODECKTWORDS, FilePathEnums.FIUDECKLABELS]
    elif language == 'FIRND':
        setup_array = [FilePathEnums.FIRNDCORPUS, FilePathEnums.FIRNDPOSSUPCORPUS, FilePathEnums.FIRNDLDMODEL,
                       FilePathEnums.FIRNDUDMODEL, FilePathEnums.FIRNDLDMAPPING, FilePathEnums.FIRNDUDMAPPING, FilePathEnums.FIRNDTWODECKTWORDS, FilePathEnums.FIRNDUDELABELS]
    else:
        print('No valid language chosen for corpus.')
    return setup_array


corpus_choices = ['FIN', 'FR', 'FIRND']
chosen_corpus = corpus_choices[2]  # Choose language.
chosen_language = chosen_corpus
chosen_corpus = corpus_instantiation(chosen_corpus)
raw_input_text = load_doc(chosen_corpus[6])
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

raw_labels = load_doc(chosen_corpus[7])
label_lines = raw_labels.split()
class_count = len(set(label_lines))
label_lines = np.array(label_lines)
output_hot = to_categorical(label_lines, class_count)
target_vector_length = len(output_hot[0])
print(output_hot.shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(word_length, vocab_size)))
model.add(tf.keras.layers.Dense(target_vector_length, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
print(model.summary())
model.compile(loss=tf.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.9, momentum=0.2),
              metrics=['accuracy'],
              )
epochs = 2000
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

model_name = ''
mapping_name = ''
if chosen_language == 'FIN':
    model_name = 'finnish_upper_deck.h5'
    mapping_name = 'finnish_upper_deck_mapping.pkl'
elif chosen_language == 'FR':
    model_name = 'french_upper_deck.h5'
    mapping_name = 'french_upper_deck_mapping.pkl'
elif chosen_language == 'FIRND':
    model_name = 'fin_random_upper_deck.h5'
    mapping_name = 'fin_random_upper_deck_mapping.pkl'
else:
    chosen_language = ''
if chosen_language:
    model.save(model_name)
    dump(mapping, open(mapping_name, 'wb'))

