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
        for subject in selected_pop:
            r = random.random()
            if r <= self.probability:
                parent = random.choice(selected_pop)
                new_population.append((subject + parent)/2)
            else:
                new_population.append(subject)
        return new_population

class Mutation():
    def __init__(self, probability):
        self.probability = probability
    
    def sum_value(self, selected_pop):
        for subject in selected_pop:
            for i, line in enumerate(subject):
                for j, value in enumerate(line):
                    r = random.random()
                    if r <= self.probability:
                        subject[i][j] = value + random.uniform(-0.1, 0.1)
                
    
    #def selection():
#    
  
if __name__ == "__main__":
    game = DinoGame(fps=1000000)
    genetic = Genetic(20)
    genetic.calc_fitness(game)
    print(genetic.get_fitness())
    
    #crossover = Crossover(0.8)
    #new_pop = crossover.mean(current_pop)