import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from pickle import load
from analytics import non_word_discrimination, single_letter_repeat, double_letter_substitution, letter_transposition, \
    relative_position_priming, transposed_letter_priming
import weight_multiplier
import output_evaluation
import tensorflow as tf
import codecs as c

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


def upper_deck_output_transcription(upper_deck_predictions):
    word_list = load_doc('../french_corpus.txt')
    word_list_lines = word_list.split()
    transcribed_outputs = list()
    for z in range(len(upper_deck_predictions)):
        transcribed_outputs.append(word_list_lines[upper_deck_predictions[z]])
    return transcribed_outputs


def two_deck(mode):
    input_output_dict = {}
    lower_deck_model = load_model('lower_deck.h5')
    lower_deck_mapping = load(open('lower_deck_mapping.pkl', 'rb'))
    if mode == "1":
        raw_input_text = load_doc('french_positional_supervised_corpus.txt')
        lower_deck_raw_input_lines = raw_input_text.split()
        lower_deck_raw_inputs = lower_deck_raw_input_lines[0:13895]  # Words change every 7 indexes.
    elif mode == "2":
        lower_deck_raw_inputs = non_word_discrimination(100, 7)
    elif mode == "3":
        lower_deck_raw_inputs = single_letter_repeat(100, 7)
    elif mode == "4":
        raw_input_text = load_doc('french_positional_supervised_corpus.txt')
        lower_deck_raw_input_lines = raw_input_text.split()
        lower_deck_raw_inputs = lower_deck_raw_input_lines[0:700]
        lower_deck_raw_inputs = lower_deck_raw_inputs[3::7]
        lower_deck_raw_inputs = double_letter_substitution(lower_deck_raw_inputs)
    elif mode == "5":
        raw_input_text = load_doc('french_positional_supervised_corpus.txt')
        lower_deck_raw_input_lines = raw_input_text.split()
        lower_deck_raw_inputs = lower_deck_raw_input_lines[0:700]
        lower_deck_raw_inputs = lower_deck_raw_inputs[3::7]
        lower_deck_raw_inputs = letter_transposition(lower_deck_raw_inputs)
    elif mode == "6":
        sub_mode_choice = input("Choose one of the following sub modes to proceed:\n"
                                "1 - Original word '1234567' changed to '1234'.\n"
                                "2 - Original word '1234567' changed to '1357'.\n")
        if sub_mode_choice != '1' and sub_mode_choice != '2':
            print("Please rerun program and choose a valid option from the prompt!")
            exit()
        raw_input_text = load_doc('french_upper_deck_inputs.txt')
        lower_deck_raw_input_lines = raw_input_text.split()
        lower_deck_raw_inputs = lower_deck_raw_input_lines[0:700]
        lower_deck_raw_inputs = lower_deck_raw_inputs[3::7]
        lower_deck_raw_inputs = relative_position_priming(lower_deck_raw_inputs, sub_mode_choice)
    elif mode == "7":
        sub_mode_choice = input("Choose one of the following sub modes to proceed:\n"
                                "1 - Original word '1234567' changed to '1235367'.\n"
                                "2 - Original word '1234567' changed to '123DD67' where 'D' is a char that was not "
                                "present in the original word.\n")
        if sub_mode_choice != '1' and sub_mode_choice != '2':
            print("Please rerun program and choose a valid option from the prompt!")
            exit()
        raw_input_text = load_doc('french_upper_deck_inputs.txt')
        lower_deck_raw_input_lines = raw_input_text.split()
        lower_deck_raw_inputs = lower_deck_raw_input_lines[0:700]
        lower_deck_raw_inputs = lower_deck_raw_inputs[3::7]
        lower_deck_raw_inputs = transposed_letter_priming(lower_deck_raw_inputs, sub_mode_choice)
    else:
        print("Please rerun program and choose a valid option from the prompt!")
        exit()
    lower_deck_vocab_size = len(lower_deck_mapping)  # Size of vocabulary
    lower_deck_word_length = 0
    lower_deck_sequences = list()
    for word in lower_deck_raw_inputs:
        if len(word) > lower_deck_word_length:  # Figure out the longest input word length. Used also for padding length if needed.
            lower_deck_word_length = len(word)
        encoded_seq = [lower_deck_mapping[char] for char in word]
        lower_deck_sequences.append(encoded_seq)
    lower_deck_sequences = np.array(lower_deck_sequences)
    lower_deck_input_hot = to_categorical(lower_deck_sequences, lower_deck_vocab_size)
    weighted_inputs = weight_multiplier.apply_input_weights(lower_deck_input_hot)
    lower_deck_outputs_str = list()
    for x in range(len(weighted_inputs)):
        lower_deck_input = weighted_inputs[x].reshape(1, lower_deck_word_length, lower_deck_vocab_size)
        lower_deck_output = lower_deck_model.predict(lower_deck_input)
        lower_deck_output = np.array(lower_deck_output)
        lower_deck_outputs_str.append(output_evaluation.output_eval(lower_deck_output))

    upper_deck_model = load_model('upper_deck.h5')
    upper_deck_mapping = load(open('upper_deck_mapping.pkl', 'rb'))
    upper_deck_vocab_size = len(upper_deck_mapping)
    upper_deck_sequences = list()
    upper_deck_outputs = list()
    upper_deck_word_length = 0
    print(lower_deck_outputs_str)
    for output_word in lower_deck_outputs_str:
        if len(output_word) > upper_deck_word_length:  # Figure out the longest input word length. Used also for padding length if needed.
            upper_deck_word_length = len(output_word)
        encoded_seq = [upper_deck_mapping[char] for char in output_word]
        upper_deck_sequences.append(encoded_seq)
    upper_deck_sequences = np.array(upper_deck_sequences)
    upper_deck_input_hot = to_categorical(upper_deck_sequences, upper_deck_vocab_size)
    for j in range(len(upper_deck_input_hot)):
        upper_deck_input = upper_deck_input_hot[j].reshape(1, upper_deck_word_length, upper_deck_vocab_size)
        upper_deck_output = upper_deck_model.predict(upper_deck_input)
        upper_deck_outputs.append(np.argmax(upper_deck_output))
    transcribed_upper_deck_outputs = upper_deck_output_transcription(upper_deck_outputs)
    if mode == 1:
        miss_predictions = {}
        for i in range(len(upper_deck_outputs)):
            input_output_dict['Raw input: ' + lower_deck_raw_inputs[i]] = 'LD output: ' + lower_deck_outputs_str[
                i], 'UDT output: ' + transcribed_upper_deck_outputs[i]
            if transcribed_upper_deck_outputs[i] not in lower_deck_raw_inputs[i]:
                miss_predictions['Raw input: ' + lower_deck_raw_inputs[i]] = 'LD output: ' + lower_deck_outputs_str[
                    i], 'UDT output: ' + transcribed_upper_deck_outputs[i]
        print(input_output_dict)
        print(
            '---------------------------------------------------------------------------------------------------------------------------------------------------------')
        print(miss_predictions)
        print('Error count: ' + str(len(miss_predictions)) + ' Total prediction count: ' + str(len(input_output_dict)))
    elif 2 <= int(mode) <= 5:  # Looking for hit rates of under 0.9. Above 0.9 indicates false positive.
        return print('tits')
    elif 6 <= int(mode) <= 7:  # Looking for hit rates of under 0.5. Above 0.5 indicates false positive.
        return print('cocks')


two_deck_mode = input("Choose one of the following modes to proceed:\n"
                      "1 - Run using the defined corpus without alterations.\n"
                      "2 - Run using Dandurand et. al. (2013) nonword evaluation.\n"
                      "3 - Run using Dandurand et. al. (2013) SRL (single repeated letter) evaluation.\n"
                      "4 - Run using Dandurand et. al. (2013) DLS (double letter substitution) evaluation.\n"
                      "5 - Run using Dandurand et. al. (2013) LT (letter transposition) evaluation.\n"
                      "6 - Run using Dandurand et. al. (2013) RPP (relative position priming) evaluation.\n"
                      "7 - Run using Dandurand et. al. (2013) TLP (transposed letter priming) evaluation.\n")

two_deck(two_deck_mode)  # Run two deck with user's chosen mode.
