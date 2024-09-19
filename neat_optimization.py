# We run this script to optimize and evolve the agent. 

# imports framework
import sys

from evoman.environment import Environment
from player_controller import player_controller
from genetic_operators import GeneticOperators

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import neat

# Define a fitness function that `neat-python` will use
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Create the environment for each genome
        env = Environment(
            experiment_name='test',
            enemies=[8],
            playermode="ai",
            player_controller=player_controller(),
            enemymode="static",
            level=2,
            logs="off", 
            savelogs="no", 
            speed="fastest",
            visuals=False
        )

        # Evaluate the genome
        fitness, _, _, _ = env.play(pcont=genome)  # Pass genome and config to player controller
        genome.fitness = fitness  # Assign the fitness score to the genome

# Load configuration for NEAT
config_path = "config"  # Create a configuration file (described below)
config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Create the population, which is the top-level object for a NEAT run
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for a maximum of 50 generations
winner = p.run(eval_genomes, 50)

# Display the winning genome
print('\nBest genome:\n{!s}'.format(winner))

