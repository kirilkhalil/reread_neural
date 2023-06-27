import codecs as c
import random
import string


def apply_filler_token(word_list):
    filler_token = '#'
    edited_word_list = list()
    for word in word_list:
        edited_word = filler_token * 3 + word + filler_token * 3
        edited_word_list.append(edited_word)
    return edited_word_list


def non_word_discrimination(word_count, letter_count):
    nonwords = list()
    letters = string.ascii_lowercase
    for i in range(word_count):
        nonwords.append(''.join(random.choice(letters) for i in range(letter_count)))
    nonwords = apply_filler_token(nonwords)
    return nonwords


def single_letter_repeat(word_count, letter_count):
    sle = list()
    letters = string.ascii_lowercase
    for i in range(word_count):
        letter_choice = random.choice(letters)
        sle.append(''.join(letter_choice for i in range(letter_count)))
    sle = apply_filler_token(sle)
    return sle


def double_letter_substitution(words, mapping):
    dls = list()
    for word in words:
        indexes = random.sample(range(3, 9), 2)
        unwanted_chars = [word[indexes[0]], word[indexes[1]]]
        banned_letters = str(unwanted_chars[0] + unwanted_chars[1])
        new_word = list(word)
        for x in range(2):
            new_word[indexes[x]] = random.choice([s for s in string.ascii_lowercase if s not in banned_letters])
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


def load_doc(filename):
    # open the file as read only
    file = c.open(filename, 'r', encoding='utf-16')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def alphabet_counter():
    raw_input_text = load_doc('french_positional_supervised_corpus.txt')
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
    # alphabet_counter()
    test = non_word_discrimination(3, 7)
    print(test)


analytics()
