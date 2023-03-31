import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


raw_text = load_doc('../word_list.rtf')
lines = raw_text.split('\n')
lines.pop()
chars = sorted(list(set(raw_text)))  # All the separate chars found in input text
chars.remove('\n')
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
vocab_size = len(mapping)  # Size of vocabulary
print(vocab_size)

#sequences = list()  # Words transposed into integer coding based on defined char values
#for line in lines:
#    encoded_seq = [mapping[char] for char in line]
#    sequences.append(encoded_seq)

vectorize_layer = layers.TextVectorization(
    standardize=None,
    split="character",
    output_mode="int",
)

vectorize_layer2 = layers.TextVectorization(
    standardize=None,
    output_mode="int",
)

# sequences = np.array(sequences)
# X, y = sequences[:, :-1], sequences[:, -1]
# sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
# X = np.array(sequences)
# y = to_categorical(y, num_classes=vocab_size)

#lines_array = np.array(lines)
vectorize_layer.adapt(lines)
#print(lines)
vec_text = vectorize_layer(lines)
lines_array = np.array(vec_text)
input_hot = to_categorical(vec_text)
#print(input_hot)
test = input_hot[0].flatten()
print(test.shape)
print(input_hot.shape)
print(input_hot.size)
#flatten = input_hot.flatten()
#print(flatten)

t_lines_array = lines
vectorize_layer2.adapt(t_lines_array)
target_vec = vectorize_layer2(t_lines_array)
t_lines_array = np.array(target_vec)
#print(t_lines_array)
target_hot = to_categorical(target_vec)
#print(target_hot.shape)
#t_flatten = target_hot.flatten()
#print(t_flatten)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(7, 28)))
model.add(tf.keras.Input(shape=196))
model.add(tf.keras.layers.Dense(502, activation='softmax'))
model.add(tf.keras.layers.Dense(502))

print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
              metrics=['accuracy'],
              )
epochs = 50
history = model.fit(input_hot, target_hot, epochs=epochs)

loss, accuracy = model.evaluate(input_hot)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

