import pygame
import random
import Brains
import math
import numpy as np
import utils

deg_to_rad = 0.0174533


def distance(pos0, pos1):
    return math.sqrt((pos0[0] - pos1[0]) ** 2 + (pos0[1] - pos1[1]) ** 2)


def population_fitness(population):
    fitness = np.array([p.eaten_count for p in population])

    if sum(fitness) > 0:
        fitness /= sum(fitness)

    return fitness


class Herbivore:

    def __init__(self, game_size, dna):
        self.position = random.randint(0, game_size[0]), random.randint(0, game_size[1])
        self.x, self.y = self.position
        self.radius = 20
        self.weights = dna[0]
        self.biases = dna[1]
        self.sight_radius = dna[2]      # the radius in which it can see food
        self.field_of_view = dna[3]     # an angle representing it's field of view

        # the two outer angles of the field of view, considering 0 to be at the creature's left
        self.left_angle = (180 - self.field_of_view) / 2
        self.right_angle = 180 - self.left_angle

        self.n_sections = 7     # number of sections the field of view is divided in, and consequently length of the input tensor
        self.section_angle = self.field_of_view / (self.n_sections - 1)

        self.speed = 0
        self.rotation = 0

        # how many frames the creature has lived
        self.frame_count = 0   # if the creature eats it is set back to zero, if it ever reaches the established lifespan, it dies

        self.eaten_count = 0    # how many meals the creature has eaten

    def update(self, food):

        visible_food = []     # the food we can actually see
        distances = []        # the distances between the visible meals and the creature
        angles = []

        for meal in food:
            d = distance(meal.position, self.position)

            x_dist = self.position[0] - meal.position[0]
            angle = math.acos((abs(x_dist)) / (d + 1e-5))
            if x_dist < 0:
                angle = 180 - angle

            if d < self.sight_radius and self.left_angle < angle < self.right_angle:
                visible_food.append(meal)
                distances.append(d)
                angles.append(angle)

        brain_input = np.zeros([1, self.n_sections])

        for i, angle in enumerate(angles):
            section = int(angle / self.section_angle)

            # ensuring we only change the value if it is 0, or if it is smaller (so the food is closer)
            if brain_input[0, section] == 0 or brain_input[0, section] > distances[i]:
                brain_input[0, section] = distances[i]

        brain_input[brain_input < 1] = 1e5      # removing all 0s from the array, they could cause confusion as in theory they are a
        # smaller distance than any other number

        out = Brains.neural_network(brain_input, self.weights, self.biases)

        self.speed = Brains.sigmoid(out[0]) * 5         # speed can range from -5 to +5
        self.rotation += Brains.sigmoid(out[1]) * 10

        """if out[1] > 5:
            self.rotation += 1
        elif out[1] < -5:
            self.rotation -= 1"""

        # moving
        self.x += math.cos(self.rotation * deg_to_rad) * self.speed
        self.y -= math.sin(self.rotation * deg_to_rad) * self.speed

        self.x = game_size[0] if self.x > game_size[0] else 0 if self.x < 0 else self.x
        self.y = game_size[1] if self.y > game_size[1] else 0 if self.y < 0 else self.y

        self.position = self.x, self.y

        self.frame_count += 1

    def render(self, surface):
        pygame.draw.circle(surface, (20, 20, 200), self.position, self.radius)
        pygame.draw.circle(surface, (255, 255, 255), self.position, int(self.sight_radius), 1)

    def reset(self, dna):
        self.eaten_count = 0
        self.frame_count = 0

        self.weights = dna[0]
        self.biases = dna[1]
        self.sight_radius = dna[2]  # the radius in which it can see food
        self.field_of_view = dna[3]  # an angle representing it's field of view


class Food:

    def __init__(self, game_size):
        self.position = random.randint(0, game_size[0]), random.randint(0, game_size[1])
        self.radius = 6

    def render(self, surface):
        pygame.draw.circle(surface, (0, 180, 0), self.position, self.radius)

    def poof(self):
        self.position = random.randint(0, game_size[0]), random.randint(0, game_size[1])


pygame.init()
FPS = 30
clock = pygame.time.Clock()

width, height = 700, 700
game_size = (width, height)
surface = pygame.display.set_mode(game_size)

food = [Food(game_size) for i in xrange(30)]

p_count = 10
herbivores_dna = utils.make_population_dna(p_count, brain_num=0)
herbivores = [Herbivore(game_size, herbivores_dna[i]) for i in xrange(p_count)]
herbivore_life_span = 120           # in frames
herbivores_fitness = population_fitness(herbivores)

play = True
while play:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break

    surface.fill((0, 0, 0))
    for meal in food:
        meal.render(surface)

    for h in herbivores:
        h.update(food)
        h.render(surface)

        if h.frame_count > herbivore_life_span:
            herbivores_fitness = population_fitness(herbivores)
            if sum(np.array(herbivores_fitness) > 0) > 2:
                i, j = np.random.choice(range(len(herbivores_dna)), 2, False, np.array(herbivores_fitness))
            else:
                i, j = np.random.choice(range(len(herbivores_dna)), 2, False)

            mates = (herbivores_dna[i], herbivores_dna[j])
            dna = utils.mutate(utils.mix(mates))
            h.reset(dna)

        for meal in food:
            if distance(meal.position, h.position) < meal.radius + h.radius:
                # food.remove(meal)
                meal.poof()
                h.eaten_count += 1
                h.frame_count = 0

    pygame.display.update()
    clock.tick(FPS)
