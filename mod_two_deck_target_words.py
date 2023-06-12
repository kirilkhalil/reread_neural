import codecs as c


def load_doc(filename):
    # open the file as read only
    file = c.open(filename, 'r', encoding='utf-16')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def save_doc(data, filename):
    file = c.open(filename, 'w', encoding='utf-16')
    file.write(data)
    file.close()


doc_loc = '../french_corpus.txt'
raw_text = load_doc(doc_loc)
words = raw_text.split()
print(len(words))
new_corpus = list()

for word in words:
    for x in range(0, 7):
        new_corpus.append(word)

save_file = " ".join(new_corpus)
print(save_file)
save_filename = 'french_two_deck_target_words.txt'
save_doc(save_file, save_filename)
