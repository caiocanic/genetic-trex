import numpy as np

from chrome_trex import DinoGame
import genetic as ga

POP_SIZE = 50
P_CROSSOVER = 0.8
P_MUTATION = 0.1
N_GENERATIONS = 100
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
    state_matrix = np.zeros((3, 30))
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
    state_matrix = np.zeros((3, 30))
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


game = DinoGame(fps=1_000_000)
#Population Initialization
population = ga.initialize_population(POP_SIZE, 30)
#First Evaluation
fitness = ga.calc_fitness(calc_fitness, population, game)
#Save first best #TODO change way the best is selected
idx_best = ga.selection.n_best(fitness, 1)
best_individual = population[idx_best]
best_fitness = fitness[idx_best]
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
    new_pop3 = ga.initialize_population(POP_SIZE//2, 30)
    new_population = np.concatenate((new_pop0, new_pop1, new_pop2, new_pop3))
    #Survivor Selection
    new_fitness = ga.calc_fitness(calc_fitness, new_population, game)
    total_population = np.concatenate((population, new_population))
    total_fitness = np.concatenate((fitness, new_fitness))
    idx_selected = ga.selection.roulette(total_fitness, POP_SIZE)
    population = total_population[idx_selected]
    fitness = total_fitness[idx_selected]
    #Save best
    idx_best = ga.selection.n_best(total_fitness, 1)
    if total_fitness[idx_best] > best_fitness:
        best_individual = total_population[idx_best]
        best_fitness = total_fitness[idx_best]
    print(gen+1, np.mean(fitness), best_fitness)