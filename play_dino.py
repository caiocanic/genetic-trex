import datetime
import numpy as np
import os

from chrome_trex import DinoGame
import genetic as ga

#TODO Try adaptative probabilities
POP_SIZE = 10
N_ATTRIBUTES = 30
P_CROSSOVER = 0.8
P_MUTATION = 1/N_ATTRIBUTES
N_GENERATIONS = 2#100
N_BEST = 10 #TODO implement N_BEST use set
N_GAMES_PLAYED = 10

#TODO Try other values as fitness: min, mean+std
def calc_fitness(subject, game):
    """
    Function used to calculate the fitness on the genetic algorithm.
    
    subject: The individual for which you want to determine the fitness.
    game: A initialized DinoGame object.
    
    return: mean scores (int). The mean of the scores of all N games
    played.
    """
    scores = []
    state_matrix = np.zeros((3, N_ATTRIBUTES))
    for _ in range(N_GAMES_PLAYED):
        game.reset()
        while not game.game_over:
            state_matrix = update_state_matrix(game, state_matrix)
            action = np.argmax(subject @ state_matrix.T)
            game.step(action)
        scores.append(game.get_score())
    return np.mean(scores)

def play_best_game(best_individual):
    """
    Play the game with the best individual obtained from the genetic
    algorithm.
    
    best_individual: Best individual after the GA has finished.
    
    return game score (int)
    """
    game = DinoGame(fps=60)
    state_matrix = np.zeros((3, N_ATTRIBUTES))
    while not game.game_over:
        state_matrix = update_state_matrix(game, state_matrix)
        action = np.argmax(best_individual @ state_matrix.T)
        game.step(action)
    return game.get_score()

def update_state_matrix(game, state_matrix):
    """
    Updated the state matrix used to obtain the action when playing the
    game by multiplying best_individual @ state_matrix.T. The state
    matrix is necessary so the dimensions match between operands.
    
    game: The DinoGame object from which the state is wanted.
    state_matrix: The previous state matrix that will be updated.
    
    return: updated state_matrix (numpy.ndarray)
    """
    state = game.get_state()
    state_matrix[0, 0:10] = state
    state_matrix[1, 10:20] = state
    state_matrix[2, 20:30] = state
    return state_matrix

def save_parameters():
    """
    Create a directory based on the current time stamp, then write the
    parameters of the genetic algorithm to a text file inside it. This
    directory will also be used to save the results from the GA.
    
    return: path save directory (str)
    """
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    save_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(save_dir)
    with open(os.path.join(save_dir, "settings.txt"), "w") as file:
        file.write(f"POP_SIZE = {POP_SIZE}\n")
        file.write(f"N_ATTRIBUTES = {N_ATTRIBUTES}\n")
        file.write(f"P_CROSSOVER = {P_CROSSOVER}\n")
        file.write(f"P_MUTATION = {P_MUTATION:.4f}\n")
        file.write(f"N_GENERATIONS = {N_GENERATIONS}\n")
        file.write(f"N_BEST = {N_BEST}\n")
        file.write(f"N_GAMES_PLAYED = {N_GAMES_PLAYED}")
    return save_dir

def save_best(population, fitness, old_best_fitness, path_best_idv):
    """
    Find the best fitness on the given population. If this population
    best fitness is greater than the one from the past generations,
    update the best individual according and save it to a npy file.
    
    population: The current population, were the best_fitness will be
    searched.
    fitness: The fitness of the individuals on the population, matched
    by the index of the array.
    old_best_fitness: The best fitness from the past generations.
    path_best_idv: Path were to save the best individual.
    
    return: max(best_fitness, old_best_fitness) (float64) Return the
    greater value between the old and new best_fitness.
    """
    idx_best = np.argmax(fitness)
    if fitness[idx_best] > old_best_fitness:
        best_fitness = fitness[idx_best]
        best_individual = population[idx_best]
        np.save(path_best_idv, best_individual)
        return best_fitness
    else:
        return old_best_fitness

#Paths
save_dir = save_parameters()    
path_pop = os.path.join(save_dir, "population.npy")
path_fitness = os.path.join(save_dir, "fitness.npy")
path_best_idv = os.path.join(save_dir, "best_individual.npy")

game = DinoGame(fps=1_000_000)
population = ga.initialize_population(POP_SIZE, N_ATTRIBUTES)
fitness = ga.calc_fitness(calc_fitness, population, game)
np.save(path_pop, population)
np.save(path_fitness, fitness)
best_fitness = save_best(population, fitness, 0, path_best_idv)
print(0, np.mean(fitness), best_fitness)
for gen in range(N_GENERATIONS):
    #Reproduction
    #   recombination and sum_value
    idx_parents = ga.selection.parents(fitness, POP_SIZE)
    new_pop0 = ga.crossover.recombination(population, P_CROSSOVER, idx_parents)
    new_pop0 = ga.mutation.sum_value(new_pop0, P_MUTATION)
    #   one_point and multiply_value
    idx_parents = ga.selection.parents(fitness, POP_SIZE)
    new_pop1 = ga.crossover.one_point(population, P_CROSSOVER, idx_parents)
    new_pop1 = ga.mutation.multiply_value(new_pop1, P_MUTATION)
    #   uniform and random resetting
    idx_parents = ga.selection.parents(fitness, POP_SIZE)
    new_pop2 = ga.crossover.uniform(population, P_CROSSOVER, idx_parents)
    new_pop2 = ga.mutation.random_resetting(new_pop2, P_MUTATION)
    #   random pop
    new_pop3 = ga.initialize_population(POP_SIZE//2, N_ATTRIBUTES)
    new_population = np.concatenate((new_pop0, new_pop1, new_pop2, new_pop3))
    #Survivor Selection
    new_fitness = ga.calc_fitness(calc_fitness, new_population, game)
    total_population = np.concatenate((population, new_population))
    total_fitness = np.concatenate((fitness, new_fitness))
    idx_selected = ga.selection.roulette(total_fitness, POP_SIZE)
    population = total_population[idx_selected]
    fitness = total_fitness[idx_selected]
    #Save population and fitness
    np.save(path_pop, population)
    np.save(path_fitness, fitness)
    #Save best individual
    best_fitness = save_best(population, fitness, best_fitness, path_best_idv)
    print(gen+1, np.mean(fitness), best_fitness)