""" This program should learn to play pong on its own. It still hasn't been tested so I don't know how long it takes to learn,
of if it even works """
from __future__ import division
import math
import random
import pygame
import numpy as np


def weight(shape):
    return np.random.normal(loc=0, scale=3, size=shape)


def bias(shape):
    return np.random.normal(loc=0, scale=2, size=shape)


def sigmoid(x):
    return 1 / (1 + math.e**-x)


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
    """ :parameter dna is a tuple with weights and biases of the current player
                    p is the percentage of mutation that will be applied """

    w, b = dna[0], dna[1]
    w += w * p / 100
    b += b * p / 100

    return w, b


def distance(pos0, pos1):
    return math.sqrt((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)


def make_population_dna(population_count, brain_num=0):
    """ returns random dna for a population of length 'population_count' """
    if brain_num == 0:
        return [([weight([2, 5]), weight([5, 1])], [bias([5]), bias([1])]) for i in xrange(population_count)]
    elif brain_num == 1:
        return [([weight([3, 6]), weight([6, 1])], [bias([6]), bias([1])]) for i in xrange(population_count)]


def make_population(game_size, p_dna):
    """ returns a population based on the dna """
    return [Player(game_size, p_dna[i]) for i in xrange(len(p_dna))]


class Player:

    def __init__(self, game_size, dna):
        """ Initializes a new player, if dna != None the weights and biases are given accordingly, else they are random
         dna = ([w0, w1], [b0, b1]) """

        print dna, '\n'
        print type(dna), '\n'
        print len(dna), '\n'

        self.x, self.y = random.randint(0, game_size[0]), game_size[1] - 100
        self.rect = pygame.Rect(self.x, self.y, 80, 10)

        self.weights = dna[0]
        self.biases = dna[1]

        self.game_width = game_size[0]
        self.points = 0

    def brain0(self, x_dist, y_dist):
        """ One of the player's possible brains. It takes as input the distances from the ball along x and y and returns either a
        positive or negative speed.
        input_shape = [1, 2] """
        input = np.reshape(np.array([x_dist, y_dist]), [1, 2]).astype(np.float32)

        layer0 = sigmoid(np.matmul(input, self.weights[0]) + self.biases[0])
        output = np.matmul(layer0, self.weights[1]) + self.biases[1]

        if output > 0:
            speed = 8
        else:
            speed = -8

        return speed

    def brain1(self, ball_x, ball_y):
        """ One of the player's possible brains. It takes as input the ball's coordinates and also the paddle's x coordinate (although
        not through the function).
        input_shape = [1, 3] """
        input = np.reshape(np.array([ball_x, ball_y, self.x]), [1, 3]).astype(np.float32)

        layer0 = sigmoid(np.matmul(input, self.weights[0]) + self.biases[0])
        output = np.matmul(layer0, self.weights[1]) + self.biases[1]

        if output > 0:
            speed = 8
        else:
            speed = -8

        return speed

    def update(self, ball_x, ball_y):
        self.rect = pygame.Rect(int(self.x), int(self.y), 80, 10)

        """x_dist = ball_x - self.x
        y_dist = self.y - ball_y
        self.x += self.brain0(x_dist, y_dist)"""

        self.x += self.brain1(ball_x, ball_y)

        # if the player goes to one edge of the screen, it will reappear on the other side, like Pac-man
        if self.x > self.game_width:
            self.x = 0
        elif self.x < 0:
            self.x = self.game_width

    def render(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), self.rect)


class AutoPlayer:

    def __init__(self, x, y, game_width):
        self.x, self.y = x, y
        self.x_speed = 10
        self.game_width = game_width
        self.rect = pygame.Rect(x, y, 80, 10)
        self.points = 0

    def update(self, ball_x):
        self.rect = pygame.Rect(self.x, self.y, 80, 10)

        if ball_x > self.x:
            self.x += self.x_speed
        elif ball_x < self.x:
            self.x -= self.x_speed

    def render(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), self.rect)


class Ball:

    def __init__(self, x, y, game_width, game_height):
        self.x, self.y = x, y
        self.game_width, self.game_height = game_width, game_height
        self.x_speed, self.y_speed = 7, 7
        self.direction = 'down'
        self.rect = pygame.Rect(x, y, 20, 20)

    def update(self):
        self.x += self.x_speed
        self.y += self.y_speed
        self.rect = pygame.Rect(self.x, self.y, 20, 20)

        if self.y > self.game_height or self.y < 0:
            self.y_speed *= -1

        elif self.x > self.game_width or self.x < 0:
            self.x_speed *= -1

        if self.y_speed > 0:
            self.direction = 'down'
        else:
            self.direction = 'up'

    def render(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), (self.x, self.y, 20, 20))


pygame.init()
FPS = 30
clock = pygame.time.Clock()
width, height = 500, 700

game_size = (width, height)
surface = pygame.display.set_mode(game_size)

player_x_speed = 0
auto_player = AutoPlayer(random.randint(0, width - 60), 100, width)
ball = Ball(350, 250, width, height)
bounced = False         # a boolean to ensure the ball doesn't get stuck on the paddles

population_count = 20
population_dna = make_population_dna(population_count, brain_num=1)
new_population_dna = []

population = make_population(game_size, population_dna)

scores = []

generation = 0
p_index = 0         # the current individual's index
while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player_x_speed = - 13
            elif event.key == pygame.K_RIGHT:
                player_x_speed = 13

            if event.key == pygame.K_SPACE:
                if FPS == 30:
                    FPS = 1e4
                else:
                    FPS = 30

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                player_x_speed = 0

    if auto_player.points >= 5:
        """ Appending the current player's score to the scores (as a measure of fitness), switching to the next player by increasing
        population_index, resetting autoplayer's points """
        print 'Populaiton index: ', p_index
        scores.append(population[p_index].points)
        p_index += 1
        auto_player.points = 0

    if p_index >= population_count:
        generation += 1
        """ one generation is over, making the next one """
        print 'Making generation', generation
        print 'Scores: ', scores
        print 'Average score: ', np.mean(np.array(scores))
        print 'Best score: ', max(scores)

        population = []
        new_population_dna = []

        while len(population) < population_count:
            i, j = np.random.choice(range(len(population_dna)), 2, False, np.array(scores) / sum(scores))
            mates = (population_dna[i], population_dna[j])
            dna = mutate(mix(mates), 2)
            new_population_dna.append(dna)
            population.append(Player(game_size, dna))

        population_dna = new_population_dna
        scores = []
        p_index = 0

    # handling hits
    if ball.rect.colliderect(population[p_index].rect) and ball.direction == 'down':
        ball.y_speed *= -1
        population[p_index].points += 1

    elif ball.rect.colliderect(auto_player.rect) and ball.direction == 'up':
        ball.y_speed *= -1
        auto_player.points += 1

    auto_player.update(ball.x)
    population[p_index].update(ball.x, ball.y)
    ball.update()

    surface.fill((230, 0, 0))
    population[p_index].render(surface)
    auto_player.render(surface)
    ball.render(surface)

    pygame.display.update()
    clock.tick(FPS)
