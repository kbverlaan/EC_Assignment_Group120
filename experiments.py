# We run this script to optimize and evolve the agent. 

import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz\bin'
os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz\bin\dot.exe'


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
def eval_genomes(genomes, config, env, experiment_name):
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
                #UNCOMMENT WHEN RGAPHVIZ WORKS
                #save_genome_plot(best_genome, config, filename=filename)

def run_experiment(experiment_name, runs, configfile, enemies, generations):
    for _ in range(runs):
        experiment_name = experiment_name

        # Create the environment for each genome
        env = Environment(
            experiment_name=experiment_name,
            enemies=enemies,
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
        config_path = configfile  # Create a configuration file (described below)
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

        winner = p.run(lambda genomes, config: eval_genomes(genomes, config, env, experiment_name), generations)
        save_winner(winner)

#number of runs
runs = 10
generations = 5

#Exploration enemy 7
print('Running experiment 1')
run_experiment('explore_7', runs, 'configexploration', [7], generations)

#Exploitation enemy 7
print('Running experiment 2')
run_experiment('exploite_7', runs, 'configexploitation', [7], generations)

#Exploration enemy 1
print('Running experiment 3')
run_experiment('explore_1', runs, 'configexploration', [1], generations)

#Exploitation enemy 1
print('Running experiment 4')
run_experiment('exploite_1', runs, 'configexploitation', [1], generations)

#Exploration enemy 2
print('Running experiment 5')
run_experiment('explore_2', runs, 'configexploration', [2], generations)

#Explotation enemy 2
print('Running experiment 6')
run_experiment('exploite_2', runs, 'configexploitation', [2], generations)