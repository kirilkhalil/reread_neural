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


def upper_deck_output_transcription(upper_deck_predictions):
    word_list = load_doc('../word_list.rtf')
    word_list_lines = word_list.split()
    transcribed_outputs = list()
    for z in range(len(upper_deck_predictions)):
        transcribed_outputs.append(word_list_lines[upper_deck_predictions[z]])
    return transcribed_outputs


input_output_dict = {}
lower_deck_model = load_model('lower_deck.h5')
lower_deck_mapping = load(open('lower_deck_mapping.pkl', 'rb'))
raw_input_text = load_doc('../positional_corpus.rtf')
lower_deck_raw_input_lines = raw_input_text.split()
lower_deck_raw_inputs = lower_deck_raw_input_lines[0:35]  # Words change every 7 indexes.
lower_deck_vocab_size = len(lower_deck_mapping)  # Size of vocabulary

lower_deck_sequences = list()
for word in lower_deck_raw_inputs:
    encoded_seq = [lower_deck_mapping[char] for char in word]
    lower_deck_sequences.append(encoded_seq)
lower_deck_sequences = np.array(lower_deck_sequences)
lower_deck_input_hot = to_categorical(lower_deck_sequences, lower_deck_vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(lower_deck_input_hot)
lower_deck_outputs_str = list()
for x in range(len(weighted_inputs)):
    lower_deck_input = weighted_inputs[x].reshape(1, 13, 27)
    lower_deck_output = lower_deck_model.predict(lower_deck_input)
    lower_deck_output = np.array(lower_deck_output)
    lower_deck_outputs_str.append(output_evaluation.output_eval(lower_deck_output))

upper_deck_model = load_model('upper_deck.h5')
upper_deck_mapping = load(open('upper_deck_mapping.pkl', 'rb'))
upper_deck_vocab_size = len(upper_deck_mapping)
upper_deck_sequences = list()
upper_deck_outputs = list()
for output_word in lower_deck_outputs_str:
    encoded_seq = [upper_deck_mapping[char] for char in output_word]
    upper_deck_sequences.append(encoded_seq)
upper_deck_sequences = np.array(upper_deck_sequences)
upper_deck_input_hot = to_categorical(upper_deck_sequences, upper_deck_vocab_size)
for j in range(len(upper_deck_input_hot)):
    upper_deck_input = upper_deck_input_hot[j].reshape(1, 7, 26)
    upper_deck_output = upper_deck_model.predict(upper_deck_input)
    upper_deck_outputs.append(np.argmax(upper_deck_output))
print(lower_deck_raw_inputs)
print(lower_deck_outputs_str)
print(upper_deck_outputs)
transcribed_upper_deck_outputs = upper_deck_output_transcription(upper_deck_outputs)
for i in range(len(upper_deck_outputs)):
    input_output_dict['Raw input: ' + lower_deck_raw_inputs[i]] = 'LD output: ' + lower_deck_outputs_str[i], 'UDS output: ' + transcribed_upper_deck_outputs[i]
print(input_output_dict)

