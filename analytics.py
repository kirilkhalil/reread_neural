import codecs as c
import random
import string


def progress_printout(ld_input, ld_output, ud_output, ud_activation_values, iteration_count):
    miss_predictions = {}
    input_output_dict = {}
    for i in range(iteration_count):
        input_output_dict['Raw input: ' + ld_input[i]] = 'LD output: ' + ld_output[
            i], 'UDT output: ' + ud_output[i], 'UD activation value: ' + str(ud_activation_values[i])
        if ud_output[i] not in ld_input[i]:
            miss_predictions['Raw input: ' + ld_input[i]] = 'LD output: ' + ld_output[
                i], 'UDT output: ' + ud_output[i], 'UD activation value: ' + str(ud_activation_values[i])
    print(input_output_dict)


def apply_filler_token(word_list, filler_count):
    filler_token = '#'
    edited_word_list = list()
    filler1 = filler_count // 2
    filler2 = filler_count - filler1
    for word in word_list:
        edited_word = filler_token * filler1 + word + filler_token * filler2
        edited_word_list.append(edited_word)
    return edited_word_list


def non_word_discrimination(word_count, letter_count, alphabet):
    nonwords = list()
    letters = ''
    for letter in alphabet:
        if letter == '#':
            continue
        letters += letter
    print(letters)
    for i in range(word_count):
        nonwords.append(''.join(random.choice(letters) for i in range(letter_count)))
    nonwords = apply_filler_token(nonwords, 6)
    return nonwords


def single_letter_repeat(word_count, letter_count, alphabet):
    sle = list()
    letters = ''
    for letter in alphabet:
        if letter == '#':
            continue
        letters += letter
    for i in range(word_count):
        letter_choice = random.choice(letters)
        sle.append(''.join(letter_choice for i in range(letter_count)))
    sle = apply_filler_token(sle, 6)
    print(sle)
    return sle


def double_letter_substitution(words, alphabet):
    dls = list()
    letters = ''
    for letter in alphabet:
        if letter == '#':
            continue
        letters += letter
    print(letters)
    for word in words:
        indexes = random.sample(range(3, 9), 2)
        unwanted_chars = [word[indexes[0]], word[indexes[1]]]
        banned_letters = str(unwanted_chars[0] + unwanted_chars[1])
        new_word = list(word)
        for x in range(2):
            new_word[indexes[x]] = random.choice([s for s in letters if s not in banned_letters])
        new_word_str = "".join(new_word)
        dls.append(new_word_str)
    return dls


def letter_transposition(words):
    lt = list()
    for word in words:
        new_word = list(word)
        new_word[7], new_word[8] = new_word[8], new_word[7]
        new_word_str = "".join(new_word)
        lt.append(new_word_str)
    return lt


def relative_position_priming(words, sub_mode):
    # If input word is 1234567 then inputs through this will change to 1234 and 1357, with rest filled with #.
    # Activation threshold is 0.5 for this test.
    rpp = list()
    for word in words:
        if sub_mode == '1':
            word = word[3:7]
        else:
            word = word[3:10:2]
        rpp.append(word)
    rpp = apply_filler_token(rpp, 9)
    return rpp


def transposed_letter_priming(words, sub_mode, alphabet):
    # If input word is 1234567 then inputs through this will be 1235467 and 123DD67, where D = char that does not
    # originally exist in the given input word.
    # Activation threshold is 0.5 for this test.
    tlp = list()
    letters = ''
    for letter in alphabet:
        if letter == '#':
            continue
        letters += letter
    for word in words:
        if sub_mode == '1':
            print(word)
            new_word = list(word)
            new_word[6], new_word[7] = new_word[7], new_word[6]
            new_word_str = "".join(new_word)
            print(new_word_str)
            tlp.append(new_word_str)
        else:
            indexes = [6, 7]
            banned_letters = list(set(word))
            new_word = list(word)

            for x in range(2):
                new_word[indexes[x]] = random.choice([s for s in letters if s not in banned_letters])
            new_word_str = "".join(new_word)
            tlp.append(new_word_str)
    return tlp


def load_doc(filename):
    # open the file as read only
    file = c.open(filename, 'r', encoding='utf-16')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def alphabet_counter():
    raw_input_text = load_doc('finnish_positional_supervised_corpus.txt')
    input_lines = raw_input_text
    print('Count of letter a: ' + str(input_lines.count('a')))
    print('Count of letter b: ' + str(input_lines.count('b')))
    print('Count of letter c: ' + str(input_lines.count('c')))
    print('Count of letter d: ' + str(input_lines.count('d')))
    print('Count of letter e: ' + str(input_lines.count('e')))
    print('Count of letter f: ' + str(input_lines.count('f')))
    print('Count of letter g: ' + str(input_lines.count('g')))
    print('Count of letter h: ' + str(input_lines.count('h')))
    print('Count of letter i: ' + str(input_lines.count('i')))
    print('Count of letter j: ' + str(input_lines.count('j')))
    print('Count of letter k: ' + str(input_lines.count('k')))
    print('Count of letter l: ' + str(input_lines.count('l')))
    print('Count of letter m: ' + str(input_lines.count('m')))
    print('Count of letter n: ' + str(input_lines.count('n')))
    print('Count of letter o: ' + str(input_lines.count('o')))
    print('Count of letter p: ' + str(input_lines.count('p')))
    print('Count of letter q: ' + str(input_lines.count('q')))
    print('Count of letter r: ' + str(input_lines.count('r')))
    print('Count of letter s: ' + str(input_lines.count('s')))
    print('Count of letter t: ' + str(input_lines.count('t')))
    print('Count of letter u: ' + str(input_lines.count('u')))
    print('Count of letter v: ' + str(input_lines.count('v')))
    print('Count of letter w: ' + str(input_lines.count('w')))
    print('Count of letter x: ' + str(input_lines.count('x')))
    print('Count of letter y: ' + str(input_lines.count('y')))
    print('Count of letter z: ' + str(input_lines.count('z')))
    print('Count of letter å: ' + str(input_lines.count('å')))
    print('Count of letter ä: ' + str(input_lines.count('ä')))
    print('Count of letter ö: ' + str(input_lines.count('ö')))
    print('Count of letter à: ' + str(input_lines.count('à')))
    print('Count of letter â: ' + str(input_lines.count('â')))
    print('Count of letter ç: ' + str(input_lines.count('ç')))
    print('Count of letter è: ' + str(input_lines.count('è')))
    print('Count of letter é: ' + str(input_lines.count('é')))
    print('Count of letter ê: ' + str(input_lines.count('ê')))
    print('Count of letter î: ' + str(input_lines.count('î')))
    print('Count of letter ï: ' + str(input_lines.count('ï')))
    print('Count of letter ô: ' + str(input_lines.count('ô')))
    print('Count of letter û: ' + str(input_lines.count('û')))
    print('Count of letter ü: ' + str(input_lines.count('ü')))


def analytics():
    alphabet_counter()

#  analytics()
