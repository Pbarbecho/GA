import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

# to reproduce the experiment
#np.random.seed(30)

##############################################################################
# Know data 
##############################################################################
hospitals = np.arange(1,11) # 10 hospitals
districts = np.arange(1,5) # 4 districs

trip_time_max = 15 # max travel time
number_of_accidents_max = 5
max_ambulances_per_hospital = 4
weight = np.random.randint(1, trip_time_max, size = len(hospitals)) # no puede ser 0 significaria que no custa nada el enlace

# weights of hospital -> districts 
W = []
for d in range(len(districts)):
    W.append(np.random.randint(1, trip_time_max, size = len(hospitals))) # no puede ser 0 


accidents = np.random.randint(1, number_of_accidents_max, size = len(hospitals))

# Constraint 
#Maximum weight that the bag of thief can hold 
ambulances_threshold = 20    

print("\nNumber of ambulances = {}".format(ambulances_threshold))

#print('Hospital   Weight(TT)     WD1            WD2          WD3         WD4         # Accidents')
print('Hospital   Weight(TT)    # Accidents(Peak hour)')
for i in range(hospitals.shape[0]):
    #print('{0}          {1}             {2}             {3}             {4}          {5}            {6}'.format(hospitals[i], weight[i], W[0][i],W[1][i],W[2][i],W[3][i], accidents[i]))
    print('{0}          {1}             {2}'.format(hospitals[i], weight[i], accidents[i]))
#######################################################################################################
# Initialize population
#######################################################################################################
solutions_per_pop = 8
pop_size = (solutions_per_pop, hospitals.shape[0])
print('Population size = {}'.format(pop_size))
initial_population = np.random.randint(5, size = pop_size) # s.t. max number of ambulances in a hospital
initial_population = initial_population.astype(int)
num_generations = 50
print('Initial population: \n{}'.format(initial_population))    


##############################################################################
# objective function / fitness function / constrains
##############################################################################
def cal_fitness(weight, accidents, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * accidents)  #Obj.
        S2 = np.sum(population[i]) # s.t.
        #S3 = np.sum(population[i] * weight[i]) # s.t.
        S4 = True # s.t. max number of ambulances per hospital
        #for x in population[i]:
        #    if x > max_ambulances_per_hospital:
        #        S4 = False
        
        if S2 <= threshold and S4:
            fitness[i] = S1
        else:    
            fitness[i] = 0 
    return fitness.astype(int) 

##############################################################################
# select parents (higher fitness values)
##############################################################################
def selection(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i,:] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999
    return parents

##############################################################################
# crossover selected parents
##############################################################################
def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    i=0
    while (parents.shape[0] < num_offsprings):
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = i%parents.shape[0]
        parent2_index = (i+1)%parents.shape[0]
        offsprings[i,0:crossover_point] = parents[parent1_index,0:crossover_point]
        offsprings[i,crossover_point:] = parents[parent2_index,crossover_point:]
        i=+1
    return offsprings   

##############################################################################
# mutation randomly flip 0 - 1
##############################################################################
def mutation(offsprings):
    mutants = np.empty((offsprings.shape))
    mutation_rate = 0.4
    for i in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[i,:] = offsprings[i,:]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0,offsprings.shape[1]-1)    
        if mutants[i,int_random_value] == 0 :
            mutants[i,int_random_value] = 1
        else :
            mutants[i,int_random_value] = 0
    return mutants 


##############################################################################
# GA main optimaze function
##############################################################################
def optimize(weight, accidents, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents 
    for i in range(num_generations):
        fitness = cal_fitness(weight, accidents, population, threshold)
        fitness_history.append(fitness)
        parents = selection(fitness, num_parents, population)
        offsprings = crossover(parents, num_offsprings)
        mutants = mutation(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
        
    print('Last generation: \n{}\n'.format(population)) 
    fitness_last_gen = cal_fitness(weight, accidents, population, threshold)      
    print('Fitness of the last generation: \n{}\n'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])
    return parameters, fitness_history


##############################################################################
# run main
##############################################################################
parameters, fitness_history = optimize(weight, accidents, initial_population, pop_size, num_generations, ambulances_threshold)
#print('The optimized parameters for the given inputs are: \n{}'.format(parameters))
print('Number of ambulances per hospital given the limited number of {} ambulances in district : \n {}'.format(ambulances_threshold, parameters))
#print(accidents)
"""
solved = []
total_weight =0
total_value =0
for i in range(selected_items.shape[1]):
    if selected_items[0][i] != 0:
        solved.append(selected_items[0][i])
        total_weight = total_weight + weight[i]
        total_value = total_value + accidents[i]
print("Total Weight = ",total_weight)
print("Total Value = ",total_value)
print("Solution: {}".format(solved))
"""


fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
fitness_history_max = [np.max(fitness) for fitness in fitness_history]
plt.plot(list(range(num_generations)), fitness_history_mean, label = 'Mean Fitness')
plt.plot(list(range(num_generations)), fitness_history_max, label = 'Max Fitness')
plt.legend()
plt.title('Fitness through the generations')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.show()
print("\n(Generations,Solutions)  ->  ", np.asarray(fitness_history).shape)








