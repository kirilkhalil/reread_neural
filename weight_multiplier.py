from enum import Enum


class FixationMultipliers(Enum):
    FIRST = [0.95, 0.65, 0.5, 0.45, 0.4, 0.35, 0.5]
    SECOND = [0.95, 0.8, 0.6, 0.5, 0.45, 0.4, 0.55]
    THIRD = [0.85, 0.75, 0.8, 0.6, 0.55, 0.5, 0.6]
    FOURTH = [0.8, 0.6, 0.7, 0.75, 0.65, 0.6, 0.65]
    FIFTH = [0.75, 0.5, 0.6, 0.7, 0.8, 0.65, 0.7]
    SIXTH = [0.65, 0.4, 0.45, 0.6, 0.7, 0.8, 0.75]
    SEVENTH = [0.6, 0.3, 0.3, 0.45, 0.55, 0.7, 0.85]


def multiplication(fixation_value, mult_values, weight_array):
    for x in range(fixation_value, fixation_value + 7):
        weight_array[x][weight_array[x].nonzero()] = mult_values[x - fixation_value]

    return weight_array


def weight_applier(nd_array, fixation):
    # Running Python 3.9 so don't have access to switch case (released in 3.10)
    if fixation == 6:  # Fixated letter = 1. Fixation-- = Fixated letter++
        multiplier = FixationMultipliers.FIRST.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    elif fixation == 5:
        multiplier = FixationMultipliers.SECOND.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    elif fixation == 4:
        multiplier = FixationMultipliers.THIRD.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    elif fixation == 3:
        multiplier = FixationMultipliers.FOURTH.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    elif fixation == 2:
        multiplier = FixationMultipliers.FIFTH.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    elif fixation == 1:
        multiplier = FixationMultipliers.SIXTH.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    elif fixation == 0:
        multiplier = FixationMultipliers.SEVENTH.value
        nd_array = multiplication(fixation, multiplier, nd_array)
    else:
        print('ERROR')

    return nd_array


def seek_fixation(array):
    non_zero_vector = array.nonzero()  # Find out which indices of the matrix contain non 0 vectors to be multiplied
    fixation = non_zero_vector[0][0]  # first index is the fixation letter of said vector
    tensor = weight_applier(array, fixation)
    return tensor


def apply_input_weights(inputs):
    for count, m_input in enumerate(inputs):  # m_input is a tensor representing one input word
        # in ohe format
        for s_count, s_input in enumerate(m_input):  # s_input is a tensor representing one input
            # letter in ohe format
            if s_input[0] == 1:  # Manually changing the ohe representation of the char '#' to be a vector of 0 bits
                s_input[0] = 0
                continue
        m_input = seek_fixation(m_input)
        # print(m_input)

    return inputs

