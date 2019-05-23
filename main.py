import numpy as np
import random

from chrome_trex import DinoGame

class Genetic():
    def __init__(self, n_population):
        self.population = []
        for _ in range(n_population):
            self.population.append(np.random.uniform(size=(3, 10)))
    
    def calc_fitness(self, game):
        self.fitness = []
        for subject in self.population:
            self.fitness.append(Genetic.__play_game(subject, game))
        
    def get_fitness(self):
        return self.fitness
        
    def get_population(self):
        return self.population
    
    def set_population(self, new_population):
        self.population = new_population
        
    def __play_game(subject, game):
        scores = []
        for _ in range(10):
            game.reset()
            while not game.game_over:
                action = np.argmax(subject @ game.get_state())
                game.step(action)
            scores.append(game.get_score())
        return np.mean(scores)
        
class Crossover():
    def __init__(self, probability):
        self.probability = probability
        
    def mean(self, selected_pop):
        assert len(selected_pop)%2 == 0
        
        new_population = []
        for i in range(0, len(selected_pop)-1, 2):
            r = random.random()
            if r <= self.probability:
                new_population.append((selected_pop[i] + selected_pop[i+1])/2)
            else:
                new_population.append(selected_pop[i])
        return new_population

#class Mutation():
#    
#        
#def selection():
#    
  
if __name__ == "__main__":
    game = DinoGame(fps=5000)
    genetic = Genetic(20)
    genetic.calc_fitness(game)
    print(genetic.get_fitness())
    
    #crossover = Crossover(0.8)
    #new_pop = crossover.mean(current_pop)