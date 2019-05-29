import numpy as np
import random

from chrome_trex import DinoGame

class Genetic():
    def __init__(self, n_population, n_attributes):
        """
        Initializes a genetic algorithm object by creating the first
        population.
        
        n_population: Size of the starting population. Number of rows
        in the population matrix.
        n_attributes: Number of attributes each individual possess.
        Number of columns on the population matrix.
        
        return: Genetic object.
        """
        self.population = np.random.uniform(size=(n_population, n_attributes))
        
    def calc_fitness(self, func, *args, **kwargs):
        """
        Calculates the fitness for each subject in the population.
        
        func: The function that will be used to calculate the fitness.
        args: Arbitrary argument lists.
        kwargs: Keyword arguments.
        
        return: None.
        """
        self.fitness = []
        for subject in self.population:
            self.fitness.append(func(subject, *args, **kwargs))
        
    def get_fitness(self):
        """
        Getter for the fitness list.
        
        return: fitness (list).
        """
        return self.fitness
        
    def get_population(self):
        """
        Getter for the population matrix.
        
        return: population (numpy.ndarray).
        """
        return self.population
    
    def set_population(self, new_population):
        """
        Override the population matrix with a new one.
        
        new_population: The new population that will replace the old
        one.
        
        return: None.
        """
        self.population = new_population
        
class Crossover():
    def __init__(self, probability):
        """
        Initializes a Crossover object, wich is responsible for applying
        the distinct types of crossover functions.
        
        probability: The probability of ocorruing a crossover.
        
        return: Crossover object
        """
        self.probability = probability
        
    def mean(self, selected_pop):
        """
        Apply crossover by the mean, where the children is the mean
        between the two parents. For each subject, another parents is
        chosen randomly.
        
        selected_pop: The selected population from wich the children
        will be generated.
        
        return: new_population (numpy.ndarray). The new population after
        the crossover.
        """
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
        """
        Initializes a Mutation object, wich is responsible for applying
        the distinct types of mutation functions.
        
        probability: The probability of ocorruing a mutation.
        
        return: Mutation object
        """
        self.probability = probability
    
    def sum_value(self, selected_pop):
        """
        Mutate the attribute of a subject by summing a value between
        [-0.1, 0.1] to it. The mutation has a chance to occur in each
        attribute from each subject.
        
        selected_pop: The selected population that will be mutated.
        
        return: new_population (numpy.ndarray). The new population after
        the mutations.
        """
        new_population = selected_pop.copy()
        for subject in new_population:
            for i, value in enumerate(subject):
                r = random.random()
                if r <= self.probability:
                    subject[i] = value + random.uniform(-0.1, 0.1)
        return new_population

class Selection():
    def __init__(self, n_selections):
        self.n_selections = n_selections
        
    def roulette(self, population, fitness):
        probabilities = fitness/sum(fitness)
        idx = np.random.choice(len(probabilities), self.n_selections,
                               p=probabilities)
        selected_pop = population[idx]
        return selected_pop
            
                
#TODO Selection also update the fitness
    #def selection():