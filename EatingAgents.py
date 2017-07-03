from __future__ import division
import pygame
import random
import math
import numpy as np

deg_to_rad = 0.0174533


def distance(pos0, pos1):
    return math.sqrt((pos0[0] - pos1[0])**2 + (pos0[1] - pos1[1])**2)


def get_angle(xy0, xy1):
    """calculates the angle between two points (starting from y=k)"""
    slope = (xy0[1] - xy1[1]) / (xy0[0] - xy1[0] + 1e-10)
    return abs(math.atan(slope) / deg_to_rad)


def make_food(food_count):
    return [Meal(random.randint(0, screen_size), random.randint(0, screen_size)) for i in xrange(food_count)]


def make_genes(population_count):
    return [np.array([random.randint(30, 40), random.randint(10, 140), random.randint(20, 300)]) for i in xrange(population_count)]


def make_population(population_count):
    return [EatingAgent(gene_pool[i], screen_size) for i in xrange(population_count)]


def mix(mates):
    result = []
    for i in xrange(len(mates[0])):
        result.append(mates[random.randint(0, 1)][i])

    return np.array(result)


def mutate(genes, p):
    mutation = p
    genes += int(random.choice([1, -1]) * mutation)
    return genes


class EatingAgent:

    """DNA: [size, sight_radius, strength]
    - the greater the size, the easier it will be to touch the food, but the heavier the agent will be
    - the sight radius is how far the agent can see, this should be as large as possible
    - strength, together with the weight, determins how fast the agent is
    - we could add: eating_speed determins how fast the agent eats, it might be too slow and some other agent might eat it once both are
    arrived at it"""

    def __init__(self, DNA, screen_size):
        self.DNA = DNA
        self.size = DNA[0] if DNA[0] >= 0 else 1
        self.sight_radius = DNA[1] if DNA[1] >= 0 else 1
        self.strength = DNA[2]

        self.x, self.y = random.randint(0, screen_size), random.randint(0, screen_size)
        self.position = self.x, self.y

        self.angle = random.randint(0, 360)     # the direction the agent is facing
        self.speed = int(self.strength / self.size)
        self.x_speed = 0
        self.y_speed = 0
        self.desired_meal = None
        self.eaten_count = 0
        self.looking_for_food = False

    def update(self, food):
        """food: the list of current food"""

        if self.desired_meal not in food:
            meal_distances = [distance(meal.position, agent.position) for meal in food]
            if min(meal_distances) <= self.sight_radius:
                self.desired_meal = food[meal_distances.index(min(meal_distances))]

                d_x, d_y = (self.desired_meal.position[0] - self.x), (self.desired_meal.position[1] - self.y)
                steps = math.sqrt((d_x**2 + d_y**2) / 25)

                self.x_speed = (self.desired_meal.position[0] - self.x) / steps
                self.y_speed = (self.desired_meal.position[1] - self.y) / steps
                self.looking_for_food = False

            elif not self.looking_for_food:
                self.x_speed = random.randint(-self.speed, self.speed)
                self.y_speed = random.randint(-self.speed, self.speed)

                self.looking_for_food = True

        self.x += self.x_speed
        self.y += self.y_speed
        self.position = self.x, self.y

        if self.x >= screen_size or self.x <= 0:
            self.x_speed *= -1

        if self.y >= screen_size or self.y <= 0:
            self.y_speed *= -1

    def render(self, surface):
        x, y = int(self.x), int(self.y)
        pygame.draw.circle(surface, (255, 255, 255), (x, y), self.sight_radius, 1)
        pygame.draw.circle(surface, (200, 0, 0), (x, y), self.size)


class Meal:

    def __init__(self, x, y):
        self.position = x, y
        self.radius = 5

    def render(self, surface):
        pygame.draw.circle(surface, (50, 200, 50), self.position, self.radius)


# setting things up
pygame.init()
FPS = 20
screen_size = 700
display = pygame.display.set_mode((screen_size, screen_size))
pygame.display.set_caption('Eating Agents')
clock = pygame.time.Clock()

population_count = 5
gene_pool = make_genes(population_count)
population = make_population(population_count)

food_count = 20
food = make_food(food_count)

new_gene_pool = []
generation_count = 0

print(mix((np.array([1, 2, 3]), np.array([4, 5, 6]))))
# game loop
play = True
while play:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            play = False
            pygame.quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # end this generation now
                food = []

    if len(food) == 0:
        new_gene_pool = []
        generation_count += 1
        print generation_count

        eaten_counts = np.array([agent.eaten_count for agent in population])
        total_eaten = sum(eaten_counts)

        while len(new_gene_pool) < population_count:
            i, j = np.random.choice(range(population_count), 2, False, eaten_counts / total_eaten)
            mates = (gene_pool[i], gene_pool[j])
            new_gene_pool.append(mutate(mix(mates), 4))

        print new_gene_pool
        gene_pool = new_gene_pool
        population = make_population(population_count)

        food = make_food(food_count)

    # rendering
    display.fill((50, 50, 150))

    for agent in population:
        agent.update(food)
        agent.render(display)

    # population[0].render(display)

    for meal in food:
        meal.render(display)

        # eating
        for agent in population:
            if distance(agent.position, meal.position) <= agent.size + meal.radius:
                agent.eaten_count += 1
                if meal in food:
                    food.remove(meal)

    pygame.display.update()
    clock.tick(FPS)
