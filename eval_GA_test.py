# -*- coding: utf-8 -*-
# eval_GA_1_test.py


import os
import random
from json import load
from src.core import route_generation, print_route, eval_GA_1
from src import utils
from deap import base, creator, tools

instName = 'A-n32-k5' #'C101' #'P-n5-k1' #'F-n135-k7'

random.seed(64)

# Customize benchmark dir location
jsonDataDir = os.path.join('benchmark', 'json_customize')
jsonFile = os.path.join(jsonDataDir, '%s.json' % instName)
with open(jsonFile) as f:
    instance = load(f)

# Create individuals
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

# Initialize values
toolbox = base.Toolbox()
IND_SIZE = 32
# Attribute generator
toolbox.register('indexes', random.sample, range(IND_SIZE), IND_SIZE)
# Structure initializers
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('select', tools.selRoulette)
toolbox.register('evaluate', eval_GA_1, instance=instance, unitCost=1.0, initCost=0)

optimalRoute = [[21, 31, 19, 17, 13, 7, 26], [12, 1, 16, 30], [27, 24], [29, 18, 8, 9, 22, 15, 10, 25, 5, 20], [14, 28, 11, 4, 23, 3, 2, 6]]

individual2 = [5, 10, 15, 22, 9, 11, 28, 18, 6, 3, 2, 17, 19, 31, 21, 13, 23, 4, 8, 29, 27, 14, 24, 30, 16, 12, 1, 7, 26, 25, 20]
route2 = route_generation(individual2, instance)

individual = [21, 31, 19, 17, 13, 7, 26, 12, 1, 16, 30, 27, 24, 29, 18, 8, 9, 22, 15, 10, 25, 5, 20, 
                14, 28, 11, 4, 23, 3, 2, 6]
route = route_generation(individual, instance)

# best_individual = [7, 19, 1, 8, 0, 11, 25, 23, 30, 12, 18, 24, 27, 2, 26, 17, 9, 6, 3, 20, 29, 28, 13, 5, 22, 21, 4, 16, 31, 10, 15, 14]
# best_route = route_generation(best_individual, instance)
best_route = [[28, 26, 29, 27], [22, 24, 23, 25], [2, 3], [16, 10, 17, 11], [4, 12, 5, 13], [18, 14, 19, 15], [8, 30, 9, 31], [6, 20, 0, 7, 1, 21]]
print('best route: %s' % best_route)
utils.visualization(instName, instance, best_route, 10, 0.8, 0.8, 50)

