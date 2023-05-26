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
save_filename = 'upper_deck_inputs.rtf'
save_labelname = 'upper_deck_labels.rtf'
save_doc(save_file, save_filename)
save_doc(label_file, save_labelname)
