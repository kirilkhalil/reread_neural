def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


raw_input_text = load_doc('../positional_corpus.rtf')
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


