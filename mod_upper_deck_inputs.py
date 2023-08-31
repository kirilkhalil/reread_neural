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


doc_loc = '../finnish_corpus.txt'
raw_text = load_doc(doc_loc)
words = raw_text.split()
word_length = 7
duplication_count = 7
class_counter = 0
new_corpus = list()
label_list = list()

for word in words:
    for x in range(0, duplication_count):
        duplicated_word = word
        new_corpus.append(duplicated_word)
        label_list.append(str(class_counter))
    class_counter += 1

save_file = " ".join(new_corpus)
label_file = " ".join(label_list)
print(label_list)
print(save_file)
save_filename = 'finnish_upper_deck_inputs.txt'
save_labelname = 'finnish_upper_deck_labels.txt'
save_doc(save_file, save_filename)
save_doc(label_file, save_labelname)
