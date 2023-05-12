import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import pad_sequences
from pickle import load
import weight_multiplier
import output_evaluation

model = load_model('lower_deck.h5')
mapping = load(open('lower_deck_mapping.pkl', 'rb'))
test_input = '######ability'

vocab_size = len(mapping)  # Size of vocabulary

sequences = list()
encoded_seq = [mapping[char] for char in test_input]
sequences.append(encoded_seq)
sequences = np.array(sequences)
input_hot = to_categorical(sequences, vocab_size)
weighted_inputs = weight_multiplier.apply_input_weights(input_hot)
print(weighted_inputs)
predict_word = model.predict(weighted_inputs)
output_evaluation.output_eval(predict_word)
