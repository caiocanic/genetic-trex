import numpy as np
import random

def initialize_population(n_population, n_attributes, low=0.0, high=1.0):
    """
    Initilizes a random population as a numpy 2D array with size
    (n_population, n_attributes). The attributes are initialized
    randomly on the range [low, high).
    
    n_population: Size of the starting population, number of rows
    in the population matrix.
    n_attributes: Number of attributes each individual possess,
    number of columns on the population matrix.
    low: The lower boundary for the attributes.
    high: The upper boundary for the attributs.
    
    return: Random population (numpy.ndarray).
    """
    return np.random.uniform(low, high, (n_population, n_attributes))
    
def calc_fitness(func, population, *args, **kwargs):
    """
    Calculates the fitness for each subject on a population according to
    the given evaluation function.
    
    func: The function that will be used to calculate the fitness.
    population: The population for which the fitness will be
    calculated.
    args: Arbitrary argument lists.
    kwargs: Keyword arguments.
    
    return: fitness array (numpy.ndarray).
    """
    fitness = np.empty(len(population))
    for i, subject in enumerate(population):
        fitness[i] = func(subject, *args, **kwargs)
    return fitness

class crossover():
    def one_point(population, probability, idx_parents):
        """
        Apply one point crossover, where a random crossover point is
        selected and the child is created by merging the head of parent
        one before that point, with the tail of parent two from that
        point and beyond. For each two selected parents, a single
        child is created.
        
        population: The population from which the children will be
        generated.
        probability: The probability of ocorruing a crossover.
        idx_parents: The list of index on the population for the
        selected parents. Each entry on the list is a tuple (p1, p2)
        representing the two parents for a single child.
        
        return: new_population (numpy.ndarray). The new population after
        the crossover.
        """
        new_population = np.empty(population.shape)
        size = population.shape[1]
        for i, (p1, p2) in enumerate(idx_parents):
            r = random.random()
            if r <= probability:
                point = random.randrange(size)
                new_population[i, :point] = population[p1, :point]
                new_population[i, point:] = population[p2, point:]
            else:
                new_population[i] = population[random.choice((p1, p2))]
        return new_population
    
    def recombination(population, probability, idx_parents):
        """
        Apply crossover by recombination, where the children are
        generated by a weighted average between the two parents. For
        each two selected parents, a single child is created.
        
        population: The population from which the children will be
        generated.
        probability: The probability of ocorruing a crossover.
        idx_parents: The list of index on the population for the
        selected parents. Each entry on the list is a tuple (p1, p2)
        representing the two parents for a single child.
        
        return: new_population (numpy.ndarray). The new population after
        the crossover.
        """        
        new_population = np.empty(population.shape)
        a = random.random()
        b = 1-a
        for i, (p1, p2) in enumerate(idx_parents):
            r = random.random()
            if r <= probability:
                new_population[i] = a*population[p1] + b*population[p2]
            else:
                new_population[i] = population[random.choice((p1, p2))]
        return new_population
        
    def uniform(population, probability, idx_parents):
        """
        Apply a uniform crossover to the population. The children are
        generated by drawing a random bit for each gene of the parents,
        if bit equals 0, the child gene comes from parent one, otherwise
        it comes from parent two. For each two selected parents, a
        single child is created.
        
        population: The population from which the children will be
        generated.
        probability: The probability of ocorruing a crossover.
        idx_parents: The list of index on the population for the
        selected parents. Each entry on the list is a tuple (p1, p2)
        representing the two parents for a single child.
        
        return: new_population (numpy.ndarray). The new population after
        the crossover.
        """
        new_population = np.empty(population.shape)
        for i, (p1, p2) in enumerate(idx_parents):
            r = random.random()
            if r <= probability:
                for j in range(new_population.shape[1]):
                    bit = random.getrandbits(1)
                    if bit == 0:
                        new_population[i, j] = population[p1, j]
                    else:
                        new_population[i, j] = population[p2, j]
            else:
                new_population[i] = population[random.choice((p1, p2))]
        return new_population

class mutation():
    def multiply_value(population, probability, a=0.9, b=1.1, low=0.0,
                       high=1.0):
        """
        Mutate the attribute of a subject by multiplying a random value
        between [a, b] to it. The value cannot go higher or lower than
        its boundaries [low, high]. The mutation has a chance to occur
        in each attribute from each subject.
        
        population: The population that will suffer mutation.
        probability: The probability of ocorruing a mutation.
        a = The start point of the random range.
        b = The end point of the random range.
        low: The lower boundary for the attributes.
        high: The upper boundary for the attributes.
        
        return: new_population (numpy.ndarray). The new population after
        the mutations.
        """
        new_population = population.copy()
        for subject in new_population:
            for i, value in enumerate(subject):
                r = random.random()
                if r <= probability:
                    subject[i] = value * random.uniform(a, b)
                    if subject[i] < low:
                        subject[i] = low
                    elif subject[i] > high:
                        subject[i] = high
        return new_population

    def random_resetting(population, probability, low=0.0, high=1.0):
        """
        Mutate the attribute of a subject by resetting it to a random
        value inside its boundaries [low, high). The mutation has a
        chance to occur in each attribute from each subject.
        
        population: The population that will suffer mutation.
        probability: The probability of ocorruing a mutation.
        low: The lower boundary for the attributes.
        high: The upper boundary for the attributes.
        
        return: new_population (numpy.ndarray). The new population after
        the mutations.
        """
        new_population = population.copy()
        for subject in new_population:
            for i in range(subject.shape[0]):
                r = random.random()
                if r <= probability:
                    subject[i] = np.random.uniform(low, high)
        return new_population    
    
    def sum_value(population, probability, a=-0.1, b=0.1, low=0.0, high=1.0):
        """
        Mutate the attribute of a subject by summing a random value
        between [a, b] to it. The value cannot go higher or lower than
        its boundaries [low, high]. The mutation has a chance to occur
        in each attribute from each subject.
        
        population: The population that will suffer mutation.
        probability: The probability of ocorruing a mutation.
        a = The start point of the random range.
        b = The end point of the random range.
        low: The lower boundary for the attributes.
        high: The upper boundary for the attributes.
        
        return: new_population (numpy.ndarray). The new population after
        the mutations.
        """
        new_population = population.copy()
        for subject in new_population:
            for i, value in enumerate(subject):
                r = random.random()
                if r <= probability:
                    subject[i] = value + random.uniform(a, b)
                    if subject[i] < low:
                        subject[i] = low
                    elif subject[i] > high:
                        subject[i] = high
        return new_population

class selection():
    def roulette(fitness, n_selected, replace=False):
        """
        Apply a roulette wheel selection to the given population.
        
        fitness: The fitness of the individuals from the population,
        matched by the index of the array.
        n_selected: The number of individuals that will be selected.
        replace: Whether the selection is with or without replacement,
        default is False.
        
        return idx_selected (numpy.ndarray). The index for the individuals
        selected by the roulette.
        """
        probabilities = fitness/sum(fitness)
        idx_selected = np.random.choice(len(probabilities), n_selected,
                                        replace=replace, p=probabilities)
        return idx_selected
    
    def n_best(fitness, n_selected):
        """
        Select the n best individuals from the population according to
        their fitness.
        
        fitness: The fitness of the individuals from the population,
        matched by the index of the array.
        n_selected: The number of individuals that will be selected.
        
        return idx_selected (numpy.ndarray). The index for the n best
        individuals.
        """
        idx_selected = np.argsort(fitness)[-n_selected:]
        return idx_selected
    
    #TODO Docstring
    #TODO Determine if this is the best way to implement parent selection
    def parents(fitness, n_selected, method="roulette"):
        """
        """
        if method == "roulette":
            idx_parents = selection._parents_roulette(fitness, n_selected)
        return idx_parents
     
    #TODO Docstring
    def _parents_roulette(fitness, n_selected):
        """
        """
        probabilities = fitness/sum(fitness)
        idx_parents = np.empty((n_selected, 2), np.int16)
        for i in range(n_selected):
            idx_parents[i] = np.random.choice(len(probabilities), (1, 2),
                                               replace=False, p=probabilities)
        return idx_parents