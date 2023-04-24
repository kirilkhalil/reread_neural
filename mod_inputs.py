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

for word in words:
    for x in range(total_length - word_length + 1):
        edited_word = filler_token * (total_length - word_length - x) + word + filler_token * x + ',' + str(class_counter)
        new_corpus.append(edited_word)
    class_counter += 1

new_corpus = str(new_corpus)
print(new_corpus)
save_filename = 'positional_supervised_corpus.rtf'
save_doc(new_corpus, save_filename)
