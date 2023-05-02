def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def save_doc(data, filename):
    file = open(filename, 'w')
    file.write(data)
    file.close()


doc_loc = '../word_list.rtf'
raw_text = load_doc(doc_loc)
words = raw_text.split()
new_corpus = list()

for word in words:
    new_corpus.append(word)

save_file = " ".join(new_corpus)
print(save_file)
save_filename = 'two_deck_target_words.rtf'
save_doc(save_file, save_filename)
