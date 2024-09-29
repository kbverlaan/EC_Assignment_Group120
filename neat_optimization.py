# We run this script to optimize and evolve the agent.

# imports framework
import sys

from evoman.environment import Environment
from player_controller import player_controller
from visualizations import save_genome_plot, plot_runs, aggregate_plots
# imports other libs
import time
import numpy as np
from math import fabs, sqrt
import glob, os
import neat
import pickle
import csv

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

# Custom class to log fitness statistics
class FitnessLogger:
    def __init__(self):
        self.generations = []
        self.max_fitnesses = []
        self.mean_fitnesses = []
        self.min_fitnesses = []

    def log_generation(self, generation, genomes):
        # genomes is a list of (genome_id, genome) tuples
        fitnesses = [genome.fitness for genome_id, genome in genomes]
        self.generations.append(generation)
        self.max_fitnesses.append(np.max(fitnesses))
        self.mean_fitnesses.append(np.mean(fitnesses))
        self.min_fitnesses.append(np.min(fitnesses))

# Number of runs
runs = 10
generations = 50

for enemy in [2, 5, 7]: 
    for run_index in range(runs):    
        experiment_name = f'CT2.0'
        run_name = f'Enemy{enemy}_Run{run_index}'

        # Create the environment for each genome
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            # multiplemode="yes",
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
        config_path = "config"  # Ensure this configuration file exists
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


        # Create an instance of FitnessLogger
        fitness_logger = FitnessLogger()
        generation_counter = [0]  # Use a list to make it mutable in nested function



        # Define eval_genomes function with access to fitness_logger and generation_counter
        def eval_genomes(genomes, config):
            # Evaluate all genomes
            for genome_id, genome in genomes:
                # Evaluate the genome
                fitness, _, _, _ = env.play(pcont=genome)  # Pass genome to player controller
                genome.fitness = fitness  # Assign the fitness score to the genome

            # After all genomes have been evaluated, sort them by fitness
            sorted_genomes = sorted(genomes, key=lambda x: x[1].fitness, reverse=True)

            # Log the fitness statistics
            fitness_logger.log_generation(generation_counter[0], genomes)

            # Get the best genome
            best_genome_id, best_genome = sorted_genomes[0]

            # Create a separate folder for the current generation
            output_dir = os.path.join(f'results/{experiment_name}/{run_name}/', f'generation_{generation_counter[0]}')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save the plot of the best genome
            filename = os.path.join(output_dir, f'best_genome_gen_{generation_counter[0]}_fitness_{round(best_genome.fitness,1)}')
            save_genome_plot(best_genome, config, filename=filename)

            # Increment generation counter
            generation_counter[0] += 1



        # Run the NEAT algorithm using our custom eval_genomes function
        winner = p.run(eval_genomes, generations)
        save_winner(winner, folder_path=f'results/{experiment_name}/{run_name}/')

        # After the run, save the collected fitness statistics
        output_dir = f'results/{experiment_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data_filename = os.path.join(output_dir, f'fitness_data_Enemy{enemy}_Run{run_index}.csv')
        with open(data_filename, 'w', newline='') as csvfile:
            fieldnames = ['generation', 'max_fitness', 'mean_fitness', 'min_fitness']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(fitness_logger.generations)):
                writer.writerow({
                    'generation': fitness_logger.generations[i],
                    'max_fitness': fitness_logger.max_fitnesses[i],
                    'mean_fitness': fitness_logger.mean_fitnesses[i],
                    'min_fitness': fitness_logger.min_fitnesses[i]
                })

        # When the run is done, create the plots using the function from visualizations
        # Assuming there is a function called create_plots in visualizations.py
        # After all runs for this enemy are completed, generate the plots
plot_runs(experiment_name, enemy, runs, generations)
aggregate_plots(experiment_name, enemy, runs, generations)