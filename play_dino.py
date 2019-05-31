import numpy as np

from chrome_trex import DinoGame
from genetic import Genetic, Crossover, Mutation, Selection

POP_SIZE = 50
P_CROSSOVER = 0.8
P_MUTATION = 0.1
N_GENERATIONS = 100
N_BEST = 10
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
       
game = DinoGame(fps=1000000)
ga = Genetic(POP_SIZE, 30)
cr = Crossover(P_CROSSOVER)
mu = Mutation(P_MUTATION)
se = Selection(POP_SIZE)
best_fitness = 0
for _ in range(N_GENERATIONS):
    #Reproduction
    #   mean and sum_value
    new_pop0 = cr.mean(ga.get_population())
    new_pop0 = mu.sum_value(new_pop0)
    #   uniform and random resetting
    new_pop1 = cr.uniform(ga.get_population())
    new_pop1 = mu.random_resetting(new_pop1,0,1)
    #   one_point and multiply_value
    new_pop2 = cr.one_point(ga.get_population())
    new_pop2 = mu.multiply_value(new_pop1)
    #   random pop #TODO Make it a function
    new_pop3 = np.random.uniform(size=(POP_SIZE//2, 30))
    total_population = np.concatenate((ga.get_population(), new_pop0,
                                       new_pop1, new_pop2, new_pop3))
    ga.set_population(total_population)
    #Evaluation
    ga.calc_fitness(calc_fitness, game)
    fitness = ga.get_fitness()
    #Selection
    #   Roulette
    idx_roulette = se.roulette(fitness)
    #   N-best
    idx_n_best = se.n_best(fitness, N_BEST)
    idx_selected = np.concatenate((idx_roulette, idx_n_best))
    ga.set_population(total_population[idx_selected])
    ga.set_fitness(fitness[idx_selected])
    #Save best
    if fitness[idx_n_best[-1]] > best_fitness:
        best_individual = total_population[idx_n_best[-1]]
        best_fitness = fitness[idx_n_best[-1]]
    print(np.mean(fitness), best_fitness)