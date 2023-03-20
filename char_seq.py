import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
np.set_printoptions(threshold=np.inf)


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load text
test_data = ["###silence###", "#talo##", '##talo#', '###talo', 'lota###', '#lota##', '##lota#', '###lota']

# Loop to create an array of Tensors to feed to NN:
data = np.fromstring(test_data[0], dtype=np.uint8) - ord('a')
print(data)

one_hot_encode = tf.one_hot(data, 26, dtype=tf.uint8)
print(one_hot_encode)
