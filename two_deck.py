import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import pad_sequences
from pickle import load
import weight_multiplier
import output_evaluation


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


lower_deck_model = load_model('lower_deck.h5')
lower_deck_mapping = load(open('lower_deck_mapping.pkl', 'rb'))
raw_input_text = load_doc('../positional_corpus.rtf')
input_lines = raw_input_text.split()
test_input = input_lines[0:1]  # Words change every 7 indexes.
vocab_size = len(lower_deck_mapping)  # Size of vocabulary

lower_deck_sequences = list()
for word in test_input:
    encoded_seq = [lower_deck_mapping[char] for char in word]
    lower_deck_sequences.append(encoded_seq)
lower_deck_sequences = np.array(lower_deck_sequences)
input_hot = to_categorical(lower_deck_sequences, vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)
for x in range(len(weighted_inputs)):
    test_input = weighted_inputs[x].reshape(1, 13, 27)
    lower_deck_output = lower_deck_model.predict(test_input)
    lower_deck_output = np.array(lower_deck_output)
    print(lower_deck_output)
    print(lower_deck_output.shape)  # Output is an array of 189 in length. Need to pick the predicted chars and ohe them again.
    outputted_word_int = output_evaluation.output_eval(lower_deck_output)
    print(outputted_word_int)  # INT representation of predicted word. Recast to ohe and feed to upper deck.
