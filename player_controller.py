# Here we create the controller for the player, so the ANN.
# When we implement NEAT, the full network structure has to be initiated here?

from evoman.controller import Controller
import numpy as np
import neat

class player_controller(Controller):
	def control(self, inputs, genome):
		config_path = "config"  # Create a configuration file (described below)
		config = neat.config.Config(
			neat.DefaultGenome,
			neat.DefaultReproduction,
			neat.DefaultSpeciesSet,
			neat.DefaultStagnation,
			config_path
		)
		# Init NN
		net = neat.nn.FeedForwardNetwork.create(genome, config)

		# Normalize inputs
		inputs = (inputs - min(inputs) / float(max(inputs) - min(inputs)))

		# Get outputs
		output = net.activate(inputs)

		# Convert output to actions
		actions = [1 if value > 0.5 else 0 for value in output]
		return actions




