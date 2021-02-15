#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:42:38 2021

@author: root
"""

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
source = np.arange(1,4)         # Distric Accident
destination =np.arange(1,5)     # Hospitals 
max_number_of_accidents = 20    # max number of accidents per districs
max_trip_time = 15              # max weight 15 minutes

# know initial  values from SUMO
sd_tupple=[]
for s_row in range(len(source)):
    for d_column in range(len(destination)):
        sd_weight = np.random.randint(1, max_trip_time)
        sd_value = np.random.randint(1, max_number_of_accidents)
        sd_tupple.append([s_row, d_column, sd_weight, sd_value])
know_data_df = pd.DataFrame(sd_tupple, columns=["Source", "Destination", "Weight", "Value"])

print(know_data_df)

# Constraint 
#Maximum number of ambulances in all districs 
max_ambulances = 30


#######################################################################################################
# Initialize population
#######################################################################################################
solutions_per_pop = 8
pop_size = (solutions_per_pop, destination.shape[0])
print('\nPopulation size = {}'.format(pop_size))
initial_population = np.random.randint(6, size = pop_size)
initial_population = initial_population.astype(int)
num_generations = 50
print('Initial population: \n{}'.format(initial_population))    

##############################################################################
# objective function / fitness function / constrains
##############################################################################
def cal_fitness(data_df, population):
            
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * value)
        S2 = np.sum(population[i] * weight)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int) 
             


