# We run this script to optimize and evolve the agent. 

# imports framework
import sys

from evoman.environment import Environment
from player_controller import player_controller
from visualizations import save_genome_plot

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import neat
import pickle

def save_winner(best_genome, folder_path='winners'):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Get the fitness of the best genome
    fitness_value = best_genome.fitness

    # Create a safe filename using the fitness value
    filename = f'best_genome_fitness_{fitness_value:.2f}.pkl'

    # Save the best genome
    with open(os.path.join(folder_path, filename), 'wb') as file:
        pickle.dump(best_genome, file)

# Define a fitness function that `neat-python` will use
def eval_genomes(genomes, config):
    best_genome = None
    best_fitness = float('-inf')

    for genome_id, genome in genomes:
        # Evaluate the genome
        fitness, _, _, _ = env.play(pcont=genome)  # Pass genome to player controller
        genome.fitness = fitness  # Assign the fitness score to the genome

        # Update the best genome if the current one is better
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome
            
            if fitness > 85:
                # Set the directory and filename for saving the plot
                output_dir = f'genomes_log/{experiment_name}'
                filename = os.path.join(output_dir, f'genome_fitness_{round(best_fitness,1)}')  # assuming env has a generation attribute

                # Save the plot of the best genome
                save_genome_plot(best_genome, config, filename=filename)

#number of runs
runs = 50
generations = 500

for _ in range(runs):
    experiment_name = 'big_test_run'

    # Create the environment for each genome
    env = Environment(
        experiment_name=experiment_name,
        enemies=[7],
        #multiplemode="yes",
        playermode="ai",
        player_controller=player_controller(),
        enemymode="static",
        level=2,
        logs="off", 
        savelogs="no", 
        speed="fastest",
        visuals=False
)
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

    winner = p.run(eval_genomes, generations)
    save_winner(winner)

