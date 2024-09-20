# We run this script to optimize and evolve the agent. 

# imports framework
import sys

from evoman.environment import Environment
from player_controller import player_controller
from Old.genetic_operators import GeneticOperators

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os


#Save results in a folder
experiment_name = 'test'
if not os.path.exists(f'results_{experiment_name}'):
    os.makedirs(f'results_{experiment_name}')

#Parameters
n_hidden_neurons = 10 #<-- switch out later
run_mode = 'train' # train or test

dom_u = 1       #<-- upper limit
dom_l = -1      #<-- lower limit
npop = 150      #<-- population size
gens = 50       #<-- number of generation
mutation = 0.2  #<-- chance of mutation


# Initialize environment
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  logs="off", 
                  savelogs="no", 
                  speed="fastest",
                  visuals=False)

env.state_to_log() # checks environment state

# Initiate time marker
ini = time.time()

# Initiate number of weights based on n_hidden_neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Initiate Genetic Operators Class
genetic_operators = GeneticOperators(env, dom_u, dom_l, npop, gens, mutation, n_vars)


# Evolution
print('START EVOLUTION')

#Init population and variables
pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
fit_pop = genetic_operators.evaluate(pop)
best = np.argmax(fit_pop)
solutions = [pop, fit_pop]
env.update_solutions(solutions)

ini_g = 0
last_sol = fit_pop[best]
notimproved = 0

print( '\n GENERATION '+str(ini_g)+' - Best Fitness '+str(round(fit_pop[best],6)))

for i in range(ini_g+1, gens):
    # Create offspring
    offspring = genetic_operators.crossover(fit_pop, pop)
    fit_offspring = genetic_operators.evaluate(offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    # Find best individual
    best = np.argmax(fit_pop)
    fit_pop[best] = float(genetic_operators.evaluate(np.array([pop[best] ]))[0])
    best_sol = fit_pop[best]

    # Select survivors
    pop, fit_pop = genetic_operators.survivor_selection(pop, fit_pop, best)


    # Doomsday event when theres no improvement in 15 generations
    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:
        pop, fit_pop = genetic_operators.doomsday(pop,fit_pop)
        print(' Doomsday Event')
        notimproved = 0


    # Print message
    best = np.argmax(fit_pop)
    print( '\n GENERATION '+str(i)+' - Best Fitness '+str(round(fit_pop[best],6)))

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

# Print total execution time
fim = time.time()
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes, ' + str(round(fim - ini)) + ' seconds\n')