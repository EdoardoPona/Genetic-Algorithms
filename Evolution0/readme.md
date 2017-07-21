Evolution0 is a genetic algorithm that evolves creatures to eat food that appear randomly around them. Each creature's dna is 
made like: (weights, biases, sight_radius, field_of_view). The last two are self explanatory, while the first two are the
variables for the creature's brain. 

- The brain:
Each creature has a neural network that controls its movements. It has as input a tensor which represents its field of view, split into as many sections as the tensor's length. Each entry is the distance from the closest meal in that section (if there
are any meals, else it is 1e5). 

At the moment it doesn't really work, but I'll leave it here while I figure out how to fix it. 
