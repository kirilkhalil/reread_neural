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
print(chars)
mapping = dict((c, i) for i, c in enumerate(chars))  # All input chars given an integer key value
vocab_size = len(mapping)  # Size of vocabulary
print(vocab_size)


vectorize_layer = layers.TextVectorization(
    standardize=None,
    split="character",
    output_mode="int",
)

vectorize_layer2 = layers.TextVectorization(
    standardize=None,
    output_mode="int",
)


vectorize_layer.adapt(lines)
print(lines)
vec_text = vectorize_layer(lines)
print(vec_text)
lines_array = np.array(vec_text)
input_hot = to_categorical(vec_text)
print(input_hot)


t_lines_array = lines
vectorize_layer2.adapt(t_lines_array)
target_vec = vectorize_layer2(t_lines_array)
t_lines_array = np.array(target_vec)
#print(t_lines_array)
target_hot = to_categorical(target_vec)
#print(target_hot.shape)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(7, 28)))
#model.add(tf.keras.layers.Dense(196))  # Likely doing sigmoid activation with fewer neurons compared to input for One-Deck
model.add(tf.keras.layers.Dense(502, activation='sigmoid'))  # Change to sigmoid

print(model.summary())
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
              metrics=['accuracy'],
              )
epochs = 100
history = model.fit(input_hot, target_hot, epochs=epochs)


print("Evaluate model on test data")
results = model.evaluate(input_hot, target_hot, batch_size=128)
print("test loss, test acc:", results)

test_input = input_hot[0].reshape(1, 7, 28)
output = model(test_input)
output = np.array(output)
print(output)
print(np.argmax(output))
