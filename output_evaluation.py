
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
    print(outputted_word)