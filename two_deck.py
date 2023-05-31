import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import pad_sequences
from pickle import load
import weight_multiplier
import output_evaluation


np.set_printoptions(threshold=np.inf)


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
print(lower_deck_mapping)
raw_input_text = load_doc('../positional_corpus.rtf')
lower_deck_raw_input_lines = raw_input_text.split()
lower_deck_raw_inputs = lower_deck_raw_input_lines[0:7]  # Words change every 7 indexes.
lower_deck_vocab_size = len(lower_deck_mapping)  # Size of vocabulary

lower_deck_sequences = list()
for word in lower_deck_raw_inputs:
    encoded_seq = [lower_deck_mapping[char] for char in word]
    lower_deck_sequences.append(encoded_seq)
lower_deck_sequences = np.array(lower_deck_sequences)
print(lower_deck_sequences)
lower_deck_input_hot = to_categorical(lower_deck_sequences, lower_deck_vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(lower_deck_input_hot)
lower_deck_outputs_str = list()
for x in range(len(weighted_inputs)):
    lower_deck_input = weighted_inputs[x].reshape(1, 13, 27)
    lower_deck_output = lower_deck_model.predict(lower_deck_input)
    lower_deck_output = np.array(lower_deck_output)
    # print(lower_deck_output)
    # print(lower_deck_output.shape)  # Output is an array of 189 in length. Need to pick the predicted chars and ohe them again.
    lower_deck_outputs_str.append(output_evaluation.output_eval(lower_deck_output))
    # print(lower_deck_outputs_int)  # INT representation of predicted word. Recast to ohe and feed to upper deck.

upper_deck_model = load_model('upper_deck.h5')
upper_deck_mapping = load(open('upper_deck_mapping.pkl', 'rb'))
print(upper_deck_mapping)
upper_deck_vocab_size = len(upper_deck_mapping)
upper_deck_sequences = list()
for output_word in lower_deck_outputs_str:
    encoded_seq = [upper_deck_mapping[char] for char in output_word]
    upper_deck_sequences.append(encoded_seq)
upper_deck_sequences = np.array(upper_deck_sequences)
print(upper_deck_sequences)
upper_deck_input_hot = to_categorical(upper_deck_sequences, upper_deck_vocab_size)
print(upper_deck_input_hot)
for j in range(len(upper_deck_input_hot)):
    upper_deck_input = upper_deck_input_hot[j].reshape(1, 7, 26)
    upper_deck_output = upper_deck_model.predict(upper_deck_input)
    print(upper_deck_output.shape)
    print(np.argmax(upper_deck_output))
