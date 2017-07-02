from __future__ import division
import pygame
import random
import numpy as np
# import scipy.signal as s
import time


"""def conv2d(x, W):
    # return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return s.convolve2d(x, W, 'same')"""


def weight(shape):
    return np.random.normal(loc=0, scale=3, size=shape)


def bias(shape):
    return np.random.normal(loc=0, scale=2, size=shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_population_dna(population_count, brain_num=0):
    """ returns random dna for a population of length 'population_count' """
    if brain_num == 0:
        return [([weight([1, 5]), weight([5, 2])], [bias([5]), bias([2])]) for i in xrange(population_count)]
    elif brain_num == 1:
        return [([weight([1, 4]), weight([4, 1])], [bias([4]), bias([1])]) for i in xrange(population_count)]


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

    # flattening all the weights and biases in the 'mates_dna' and adding them to a new list
    weights = []
    biases = []

    n_of_vars = len(mates_dna[0][0])            # the number of weights of the first parent, which corresponds to the number of biases

    for i in xrange(n_of_vars):
        parent = mates_dna[random.randint(0, len(mates_dna) - 1)]

        weights.append(parent[0][i])
        biases.append(parent[1][i])

    return np.array(weights), np.array(biases)


def mutate(dna, p):
    """dna is a tuple with weights and biases of the current player
        p is the percentage of mutation that will be applied """
    w, b = dna[0], dna[1]
    w += w * p / 100
    b += b * p / 100

    return w, b


class Player:

    def __init__(self, dna, x=300, y=300):
        self.rect = pygame.Rect(x, y, 40, 40)
        self.y_speed = 0
        self.is_grounded = True
        self.x = x

        self.weights = dna[0]
        self.biases = dna[1]

    def update(self, distance):
        x, y = self.rect[0], self.rect[1]
        y += self.y_speed

        if y >= 300 - 40:
            y = 300 - 40
            self.is_grounded = True

        if not self.is_grounded:
            self.y_speed += 3

        self.brain1(distance)

        self.rect = pygame.Rect(x, y, 40, 40)

    def brain0(self, distance):
        input = np.reshape(np.array([distance]), [1, 1]).astype(np.float32)

        layer0 = sigmoid(np.matmul(input, self.weights[0]) + self.biases[0])
        output = np.matmul(layer0, self.weights[1]) + self.biases[1]
        output = np.reshape(output, [2])

        if output[0] > output[1]:
            self.jump()

    def brain1(self, distance):
        input = np.reshape(np.array([distance]), [1, 1]).astype(np.float32)

        layer0 = sigmoid(np.matmul(input, self.weights[0]) + self.biases[0])
        output = np.matmul(layer0, self.weights[1]) + self.biases[1]
        output = np.reshape(output, [1])

        if output[0] > 0:
            self.jump()

    def render(self, surface):
        pygame.draw.rect(surface, (0, 0, 0), self.rect)

    def jump(self):
        if self.is_grounded:
            self.y_speed = -30
            self.is_grounded = False

    def set_genes(self, dna):
        self.weights = dna[0]
        self.biases = dna[1]


class Block:

    def __init__(self, x, y, player_speed):
        self.rect = (x, y, 40, 70)
        self.speed = -player_speed
        self.x = x

    def update(self):
        self.x += self.speed

        if self.x <= 0:
            self.reset_x()

        self.rect = pygame.Rect(self.x, self.rect[1], 40, 70)

    def render(self, surface):
        pygame.draw.rect(surface, (0, 0, 0), self.rect)

    def reset_x(self):
        self.x = random.randint(700, 1500)


pygame.init()
FPS = 30
clock = pygame.time.Clock()
width, height = 700, 500

game_size = (width, height)
surface = pygame.display.set_mode(game_size)
floor = 300

p_count = 10
population_dna = make_population_dna(p_count, brain_num=1)

player = Player(population_dna[0], y=floor-40)
block = Block(1000, floor-70, 10)
parent_num = 2

scores = [0]
index = 0
generation = 0
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                player.jump()

            if event.key == pygame.K_SPACE:
                if FPS == 30:
                    FPS = 1e4
                else:
                    FPS = 30

    # update
    player.update(block.x - player.x)
    block.update()

    # render
    surface.fill((255, 255, 255))
    player.render(surface)
    block.render(surface)

    if block.rect.colliderect(player.rect):
        print 'Individual ', index, ' scored ', scores[index]
        index += 1
        scores.append(0)
        block.reset_x()
        player.set_genes(population_dna[index])

    elif abs(block.x - player.x) < 5:
        scores[index] += 1

    if index == p_count - 1:
        generation += 1
        index = 0
        scores = np.array(scores)
        print 'Scores: ', scores
        print 'Average score: ', np.mean(np.array(scores))
        print 'Best score: ', max(scores)

        print 'Making generation', generation

        new_population_dna = []
        while len(new_population_dna) < p_count:

            if sum(scores != 0) > parent_num:
                i, j = np.random.choice(range(p_count), parent_num, False, scores / sum(scores))
            else:
                i, j = np.random.choice(range(p_count), parent_num)

            mates = (population_dna[i], population_dna[j])
            dna = mutate(mix(mates), 2)
            new_population_dna.append(dna)

        population_dna = new_population_dna
        new_population_dna = []
        scores = [0]

    pygame.display.update()
    clock.tick(FPS)
