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


experiment_name = 'winner_demo'

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
    speed='normal',
    visuals=True
)

# Ask the user for the filename
file_name = 'best_genome_fitness_89.37'

# Define the path to the file
file_path = f'winners/{file_name}.pkl'

# Open and load the pickle file
with open(file_path, 'rb') as file:
    genome = pickle.load(file)

env.play(pcont=genome)