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
total_length = 13
class_counter = 1
filler_token = '#'
new_corpus = list()
label_list = list()

for word in words:
    for x in range(total_length - word_length + 1):
        edited_word = filler_token * (total_length - word_length - x) + word + filler_token * x
        new_corpus.append(edited_word)
        label_list.append(str(class_counter))
    class_counter += 1

save_file = " ".join(new_corpus)
label_file = " ".join(label_list)
print(label_list)
print(save_file)
save_filename = 'positional_supervised_corpus.rtf'
save_labelname = 'labels.rtf'
save_doc(save_file, save_filename)
save_doc(label_file, save_labelname)
