import numpy as np

from chrome_trex import DinoGame
from genetic import Genetic, Crossover, Mutation, Selection

POP_SIZE = 50
P_CROSSOVER = 0.8
P_MUTATION = 0.1
N_GENERATIONS = 100

def calc_fitness(subject, game):
    scores = []
    for _ in range(10):
        game.reset()
        while not game.game_over:
            state_matrix = create_state_matrix(game)
            action = np.argmax(subject @ state_matrix.T)
            game.step(action)
        scores.append(game.get_score())
    return np.mean(scores)

def create_state_matrix(game):
    state = game.get_state()
    state_matrix = np.zeros((3, 30))
    a = 0
    b = 10 
    for i in range(3):
        state_matrix[i, a:b] = state
        a = b
        b = b+b
    return state_matrix
    
def play_game(best_individual):
    game = DinoGame(fps=60)
    while not game.game_over:
        state_matrix = create_state_matrix(game)
        action = np.argmax(best_individual @ state_matrix.T)
        game.step(action)
    return game.get_score()
       
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
    idx_n_best = se.n_best(fitness, 5)
    idx_selected = np.concatenate((idx_roulette, idx_n_best))
    ga.set_population(total_population[idx_selected])
    ga.set_fitness(fitness[idx_selected])
    #Save best
    if fitness[idx_n_best[-1]] > best_fitness:
        best_individual = total_population[idx_n_best[-1]]
        best_fitness = fitness[idx_n_best[-1]]
    print(np.mean(fitness), best_fitness)