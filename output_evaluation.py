def int_to_char(word):
    output_word = ''
    for char_num in word:
        # print(char_num)
        output_word += chr(ord('`') + char_num)
    print(output_word)
    return output_word


def char_eval(candidates):
    chosen_candidate = 0
    chosen_index = 0
    for count, candidate in enumerate(candidates):  # Look for highest value from 27 char set and return its index
        if candidate > chosen_candidate:
            chosen_candidate = candidate
            chosen_index = count
    return chosen_index


def output_eval(raw_output):
    outputted_word = list()
    char_list = list()
    char_splitter = 0
    split_value = [26, 53, 80, 107, 134, 161, 188]
    for x in range(0, 189):
        char_list.append(raw_output[0][x])  # Add all char candidates in 27 char intervals
        if char_splitter in split_value:
            outputted_word.append(char_eval(char_list))
            char_list = list()
        char_splitter += 1
    return int_to_char(outputted_word)


# First prio to create actual two deck
# Use French lexicon
# Make a function to count the number of alphabet instances in the input file (i.e how many letters 'a' in input words)
# Make a function to count the number of alphabet instances in specific position of input file (i.e how many letters 'a' in position 1, position 2 etc)
# Find out if we can take snapshots from a single training run (max 30k runs and we take snapshots from 5,10,100 epochs etc.
# Sum of correct/incorrect predictions. Also able to check what specific word is hard.

