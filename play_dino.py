import numpy as np

from chrome_trex import DinoGame
from genetic import Genetic, Crossover, Mutation, Selection

POP_SIZE = 25
P_CROSSOVER = 0.8
P_MUTATION = 0.1
N_GENERATIONS = 50

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
    
def play_game(game, best_individual):
    game.reset()
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
    new_population = cr.mean(ga.get_population())
    new_population = mu.sum_value(new_population)
    total_population = np.concatenate((ga.get_population(), new_population))
    ga.set_population(total_population)
    #Evaluation
    ga.calc_fitness(calc_fitness, game)
    fitness = ga.get_fitness()
    #Save best
    idx_best = ga.best_individual()
    if fitness[idx_best] > best_fitness:
        best_individual = total_population[idx_best]
        best_fitness = fitness[idx_best]
    #Selection
    idx_selected = se.roulette(total_population, fitness)
    ga.set_population(total_population[idx_selected])
    ga.set_fitness(fitness[idx_selected])
    print(np.mean(fitness), best_fitness)