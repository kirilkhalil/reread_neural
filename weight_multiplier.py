from enum import Enum


class FixationMultipliers(Enum):
    FIRST = [0.95, 0.65, 0.5, 0.45, 0.4, 0.35, 0.5]


def weight_applier(nd_array, fixation):
    # print(nd_array)
    # Running Python 3.9 so don't have access to switch case (released in 3.10)
    if fixation == 6:  # Fixated letter = 1. Fixation-- = Fixated letter++
        multiplier = FixationMultipliers.FIRST.value
        for x in range(fixation, fixation+7):
            # print(x)
            print(nd_array[x][1])  # Figure out how to invoke the right index to multiply with values
            # nd_array[x][1] = nd_array[x][1] * multiplier[x-6]
            # print(nd_array[x])
        print('test 6')
    elif fixation == 5:
        print('test 5')
    elif fixation == 4:
        print('test 4')
    elif fixation == 3:
        print('test 3')
    elif fixation == 2:
        print('test 2')
    elif fixation == 1:
        print('test 1')
    elif fixation == 0:
        print('test 0')
    else:
        print('ERROR')

    return nd_array


def seek_fixation(array):
    # print(tensor)
    non_zero_vector = array.nonzero()  # Find out which indices of the matrix contain non 0 vectors to be multiplied
    fixation = non_zero_vector[0][0]  # first index is the fixation letter of said vector
    tensor = weight_applier(array, fixation)
    # print(fixation_point)
    # print(tensor[6])

    return tensor


def apply_input_weights(inputs):
    for count, m_input in enumerate(inputs):  # m_input is a tensor the shape of (13, 27) representing one input word
        # in ohe format
        for s_count, s_input in enumerate(m_input):  # s_input is a tensor the shape of (27,0) representing one input
            # letter in ohe format
            # print(s_count)
            if s_input[0] == 1:  # Manually changing the ohe representation of the char '#' to be a vector of 0 bits
                s_input[0] = 0
                continue
        m_input = seek_fixation(m_input)
        # print(m_input)

    # print(inputs[0])
    return inputs


# Pattern that repeats for letter positions:
# (array([ 6,  7,  8,  9, 10, 11, 12]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([ 5,  6,  7,  8,  9, 10, 11]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([ 4,  5,  6,  7,  8,  9, 10]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([3, 4, 5, 6, 7, 8, 9]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([2, 3, 4, 5, 6, 7, 8]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([1, 2, 3, 4, 5, 6, 7]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([0, 1, 2, 3, 4, 5, 6]), array([ 1,  2,  9, 12,  9, 20, 25]))
# (array([ 6,  7,  8,  9, 10, 11, 12]), array([ 1,  2,  9, 12,  9, 20, 25]))
# ETC.....
