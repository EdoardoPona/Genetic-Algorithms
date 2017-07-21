import numpy as np
import random


def weight(shape):
    return np.random.uniform(-5, 5, shape)


def bias(shape):
    return np.random.normal(-5, 5, shape)


def flatten(x):
    """ returns a 1 dim array from the input """
    length = 1
    for i in x.shape:
        length *= i

    return np.reshape(x, [length])


def mix_array(arrays):
    """randomly mixes the arrays in 'arrays' along all their dimensions"""

    new_array = []

    flat_arrays = [flatten(i) for i in arrays]
    for i in xrange(len(flat_arrays[0])):
        new_array.append(random.choice(flat_arrays)[i])

    return np.reshape(np.array(new_array), arrays[0].shape)


def mix(mates_dna):
    """
    :parameter
     mates_dna is a list with the weights and biases (contained in tuples) of the mating paddles
    Eg: [([w0_0, w1_0, ...], [b0_0, b1_0, ...]), ([w0_1, w1_1, ...], [b0_1, b1_1, ...]), ...]
    mates_dna: (parent0, parent1, ...)
    parent: (weights, biases)
    weights, biases = [w0, w1, ...], [b0, b1, ...]
    :return
        a tuple with mixed weights and biases """

    # weights
    new_weights = []
    current_weights = []

    for i in xrange(len(mates_dna[0][0])):
        for parent_inedx in xrange(len(mates_dna)):
            current_weights.append(mates_dna[parent_inedx][0][i])

        new_weights.append(mix_array(current_weights))
        current_weights = []

    # biases
    new_biases = []
    current_biases = []

    for i in xrange(len(mates_dna[0][1])):
        for parent_inedx in xrange(len(mates_dna)):
            current_biases.append(mates_dna[parent_inedx][1][i])

        new_biases.append(mix_array(current_biases))
        current_biases = []

    new_sight_radius = random.choice([mates_dna[0][2], mates_dna[1][2]])
    new_field_of_view = random.choice([mates_dna[0][3], mates_dna[1][3]])

    return np.array(new_weights), np.array(new_biases), new_sight_radius, new_field_of_view


def mutate(dna):
    """dna is a tuple with weights and biases of the current player
        p is the percentage of mutation that will be applied """

    weights, biases = dna[0], dna[1]
    new_weights = []
    new_biases = []

    for w in weights:
        new_weight = []
        for val in flatten(w):
            if random.choice(range(0, 100)) in range(100):
                new_weight.append(np.random.uniform(-5, 5))
            else:
                new_weight.append(val)

        new_weights.append(np.reshape(np.array(new_weight), w.shape))

    new_weights = np.reshape(np.array(new_weights), weights.shape)

    for b in biases:
        new_bias = []
        for val in flatten(b):
            if random.choice(range(0, 100)) in range(100):
                new_bias.append(np.random.uniform(-5, 5))
            else:
                new_bias.append(val)

        new_biases.append(np.reshape(np.array(new_bias), b.shape))

    new_biases = np.reshape(np.array(new_biases), weights.shape)

    new_sight_radius = dna[2] * random.choice([0.99, 1.01f])
    new_field_of_view = dna[3] * random.choice([0.99, 1.01])

    return new_weights, new_biases, new_sight_radius, new_field_of_view


def make_population_dna(population_count, brain_num=0):
    """ returns random dna for a population of length 'population_count' """
    if brain_num == 0:
        return [([weight([7, 10]), weight([10, 8]), weight([8, 2])], [bias([10]), bias([8]), bias([2])],    # weights and biases
                 random.randint(2, 70),     # sight radius
                random.randint(2, 180))        # field of view
                for i in xrange(population_count)]
