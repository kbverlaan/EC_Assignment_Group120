import warnings
import graphviz
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology, ensuring inputs are always at the same place. """
    
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    # Fix the input node positions
    input_layer_y = 0  # Define a constant y-coordinate for all input nodes
    input_step = 2  # Horizontal distance between input nodes

    inputs = set()
    for i, k in enumerate(config.genome_config.input_keys):
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        
        # Set the position using the pos attribute
        x_pos = i * input_step
        dot.node(name, _attributes={**input_attrs, 'pos': f'{x_pos},{input_layer_y}!'})

    # Fix the output node positions
    output_layer_y = -4  # Define a constant y-coordinate for all output nodes
    output_step = 2  # Horizontal distance between output nodes

    outputs = set()
    for i, k in enumerate(config.genome_config.output_keys):
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        
        # Set the position using the pos attribute
        x_pos = i * output_step
        dot.node(name, _attributes={**node_attrs, 'pos': f'{x_pos},{output_layer_y}!'})

    # Add hidden nodes
    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    # Add connections between nodes
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

def save_genome_plot(genome, config, filename='genome_plot', node_names=None, show_disabled=True, prune_unused=False, fmt='svg'):
    """ Draws and saves the neural network of a given genome. """
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Draw the network using the draw_net function and save it to the specified file
    draw_net(config, genome, view=False, filename=filename, node_names=node_names, show_disabled=show_disabled, prune_unused=prune_unused, fmt=fmt)

def plot_runs(experiment_name, enemy, runs, generations):
    # Lists to store data from all runs
    all_max_fitnesses = []
    all_mean_fitnesses = []
    generations_list = None

    for run_index in range(runs):
        data_filename = os.path.join('results', experiment_name, f'fitness_data_Enemy{enemy}_Run{run_index}.csv')

        # Read the fitness data CSV file
        if os.path.exists(data_filename):
            df = pd.read_csv(data_filename)
            all_max_fitnesses.append(df['max_fitness'].values)
            all_mean_fitnesses.append(df['mean_fitness'].values)
            if generations_list is None:
                generations_list = df['generation'].values
        else:
            print(f'Fitness data file not found for run {run_index}')
            continue

    # Now plot all runs in a single plot
    plt.figure(figsize=(10, 6))

    # Plot best fitness lines for each run
    for idx, max_fitnesses in enumerate(all_max_fitnesses):
        plt.plot(generations_list, max_fitnesses, linestyle='-', color='blue', alpha=0.5)

    # Plot mean fitness lines for each run
    for idx, mean_fitnesses in enumerate(all_mean_fitnesses):
        plt.plot(generations_list, mean_fitnesses, linestyle='--', color='orange', alpha=0.5)

    # Add labels for best and mean fitness
    plt.plot([], [], linestyle='-', color='blue', label='Best Fitness per Run')
    plt.plot([], [], linestyle='--', color='orange', label='Mean Fitness per Run')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    #plt.title(f'Fitness over Generations for Enemy {enemy} with Compatibility Threshold = 3.0')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the plot
    output_dir = os.path.join('results', experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'fitness_over_generations_enemy_{enemy}.png'))
    plt.close()

def aggregate_plots(experiment_name, enemy, runs, generations):
    # Lists to store data from all runs
    all_max_fitnesses = []
    all_mean_fitnesses = []
    generations_list = None

    for run_index in range(runs):
        data_filename = os.path.join('results', experiment_name, f'fitness_data_Enemy{enemy}_Run{run_index}.csv')

        # Read the fitness data CSV file
        if os.path.exists(data_filename):
            df = pd.read_csv(data_filename)
            all_max_fitnesses.append(df['max_fitness'].values)
            all_mean_fitnesses.append(df['mean_fitness'].values)
            if generations_list is None:
                generations_list = df['generation'].values
        else:
            print(f'Fitness data file not found for run {run_index}')
            continue

    # Convert lists to numpy arrays
    all_max_fitnesses = np.array(all_max_fitnesses)
    all_mean_fitnesses = np.array(all_mean_fitnesses)

    # Compute mean and std deviation across runs
    mean_max_fitnesses = np.mean(all_max_fitnesses, axis=0)
    std_max_fitnesses = np.std(all_max_fitnesses, axis=0)

    mean_mean_fitnesses = np.mean(all_mean_fitnesses, axis=0)
    std_mean_fitnesses = np.std(all_mean_fitnesses, axis=0)

    print(f'Enemy {enemy}')
    print(mean_max_fitnesses[-1])
    print(std_max_fitnesses[-1])
    print(max([item for sublist in all_max_fitnesses for item in sublist]))
    

    # Plot the aggregated data
    plt.figure(figsize=(10, 6))

    # Plot mean of best fitness with std deviation
    plt.plot(generations_list, mean_max_fitnesses, label='Mean Best Fitness', linestyle='-', color='blue')
    plt.fill_between(generations_list, mean_max_fitnesses - std_max_fitnesses, mean_max_fitnesses + std_max_fitnesses, color='blue', alpha=0.2)

    # Plot mean of mean fitness with std deviation
    plt.plot(generations_list, mean_mean_fitnesses, label='Mean of Mean Fitness', linestyle='--', color='orange')
    plt.fill_between(generations_list, mean_mean_fitnesses - std_mean_fitnesses, mean_mean_fitnesses + std_mean_fitnesses, color='orange', alpha=0.2)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    #plt.title(f'Aggregated Fitness over Generations for Enemy {enemy} with Compatibility Threshold = 3.0')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save the plot
    output_dir = os.path.join('results', experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'aggregated_fitness_enemy_{enemy}.png'))
    plt.close()