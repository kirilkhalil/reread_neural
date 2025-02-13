import codecs as c
from strenum import StrEnum


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


class FilePathEnums(StrEnum):
    FRCORPUS = 'french_corpus.txt'
    FICORPUS = 'finnish_corpus.txt'
    FIRANDOMTESTCORPUS = 'fin_random_corpus.txt'


def corpus_instantiation(language):  # Add cases as required for new language options. Make sure new entries follow the same ordering!
    setup_array = []
    if language == 'FIN':
        setup_array = [FilePathEnums.FICORPUS]
    elif language == 'FR':
        setup_array = [FilePathEnums.FRCORPUS]
    elif language == 'FIRNDTEST':
        setup_array = [FilePathEnums.FIRANDOMTESTCORPUS]
    else:
        print('No valid language chosen for corpus.')
    return setup_array


corpus_choices = ['FIN', 'FR', 'FIRNDTEST']
chosen_corpus = corpus_choices[2]  # Choose language.
chosen_language = chosen_corpus
chosen_corpus = corpus_instantiation(chosen_corpus)
doc_loc = '../' + chosen_corpus[0]
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

target_name = ''
label_name = ''
if chosen_language == 'FIN':
    target_name = 'finnish_two_deck_target_words.txt'
    label_name = 'finnish_upper_deck_labels.txt'
elif chosen_language == 'FR':
    target_name = 'french_two_deck_target_words.txt'
    label_name = 'french_upper_deck_labels.txt'
elif chosen_language == 'FIRNDTEST':
    target_name = 'fin_random_two_deck_target_words.txt'
    label_name = 'finnish_random_upper_deck_labels.txt'
else:
    chosen_language = ''
if chosen_language:
    save_doc(save_file, target_name)
    save_doc(label_file, label_name)
