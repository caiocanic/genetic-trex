import numpy as np

from chrome_trex import DinoGame
from genetic import Genetic, Crossover, Mutation, Selection

POP_SIZE = 50
P_CROSSOVER = 0.8
P_MUTATION = 0.1
N_GENERATIONS = 100
N_BEST = 5
N_GAMES_PLAYED = 10

#TODO Try other values as fitness: min, mean+std
def calc_fitness(subject, game):
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

def play_game(best_individual):
    game = DinoGame(fps=60)
    state_matrix = np.zeros((3, 30))
    while not game.game_over:
        state_matrix = update_state_matrix(game, state_matrix)
        action = np.argmax(best_individual @ state_matrix.T)
        game.step(action)
    return game.get_score()

def update_state_matrix(game, state_matrix):
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
    total_population = np.concatenate((ga.get_population(), new_pop0,
                                       new_pop1))
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