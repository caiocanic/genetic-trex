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
    
game = DinoGame(fps=1000000)
ga = Genetic(POP_SIZE, 30)
cr = Crossover(P_CROSSOVER)
mu = Mutation(P_MUTATION)
se = Selection(POP_SIZE)
for _ in range(N_GENERATIONS):
    new_population = cr.mean(ga.get_population())
    new_population = mu.sum_value(new_population)
    total_population = np.concatenate((ga.get_population(), new_population))
    ga.set_population(total_population)
    ga.calc_fitness(calc_fitness, game)
    fitness = ga.get_fitness()
    selected_pop = se.roulette(total_population, fitness)
    ga.set_population(selected_pop)
    print(np.mean(fitness))