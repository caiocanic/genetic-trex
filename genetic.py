import numpy as np
import random

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
        self.fitness = np.empty(len(self.population))
        for i, subject in enumerate(self.population):
            self.fitness[i] = func(subject, *args, **kwargs)
        
    def best_individual(self):
        """
        Find the index to the best individual (higher fitness) in the
        population.
        
        return: idx_best (int)
        """
        idx_best = np.argmax(self.fitness)
        return idx_best
    
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
    
    
    def set_fitness(self, new_fitness):
        """
        Override the fitness array with a new one.
        
        new_fitness: The new fitness array that will replace the old
        one.
        
        return: None
        """
        self.fitness = new_fitness
    
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
        Initializes a Crossover object, which is responsible for applying
        the distinct types of crossover functions.
        
        probability: The probability of ocorruing a crossover.
        
        return: Crossover object.
        """
        self.probability = probability
        
    def mean(self, population):
        """
        Apply crossover by the mean, where the children is the mean
        between the two parents. For each subject, another parents is
        chosen randomly.
        
        population: The population from which the children will be
        generated.
        
        return: new_population (numpy.ndarray). The new population after
        the crossover.
        """
        new_population = np.empty(population.shape)
        for i, subject in enumerate(population):
            r = random.random()
            if r <= self.probability:
                parent = random.choice(population)
                new_population[i] = (subject + parent)/2
            else:
                new_population[i] = subject
        return new_population

class Mutation():
    def __init__(self, probability):
        """
        Initializes a Mutation object, which is responsible for applying
        the distinct types of mutation functions.
        
        probability: The probability of ocorruing a mutation.
        
        return: Mutation object.
        """
        self.probability = probability
    
    def sum_value(self, population):
        """
        Mutate the attribute of a subject by summing a value between
        [-0.1, 0.1] to it. The mutation has a chance to occur in each
        attribute from each subject.
        
        population: The population that will suffer mutation.
        
        return: new_population (numpy.ndarray). The new population after
        the mutations.
        """
        new_population = population.copy()
        for subject in new_population:
            for i, value in enumerate(subject):
                r = random.random()
                if r <= self.probability:
                    subject[i] = value + random.uniform(-0.1, 0.1)
        return new_population

class Selection():
    def __init__(self, n_selected):
        """
        Initializes a Selection object, which is responsible for applying
        the distinct types of selection functions.
        
        n_selected: The number of individuals that will be selected.
        
        return: Selection object.
        """
        self.n_selected = n_selected
        
    def roulette(self, population, fitness):
        """
        Apply a roulette wheel selection to the given population.
        
        population: The population from which the individuals will be
        selected.
        fitness: The fitness of the individuals from the population,
        matched by the index of the list.
        
        return idx_selected (numpy.ndarray). The index for the individuals
        selected by the roulette.
        """
        probabilities = fitness/sum(fitness)
        idx_selected = np.random.choice(len(probabilities), self.n_selected,
                               p=probabilities)
        return idx_selected