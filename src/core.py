# -*- coding: utf-8 -*-
# core.py

import os
import random
import numpy
from json import load
import csv
from deap import base, creator, tools
from CRZ_profile import profile
from .utils import makeDirsForFile, exist
from . import BASE_DIR
from numba import jit
from itertools import takewhile
# BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def print_route(route, merge=False):
    routeStr = ''
    subRouteCount = 0
    for subRoute in route:
        subRouteCount += 1
        subRouteStr = ''

        for customerID in subRoute:
            subRouteStr = subRouteStr + ' - ' + str(customerID)
            routeStr = routeStr + ' - ' + str(customerID)
        if not merge:
            print('  Vehicle %d\'s route: %s' % (subRouteCount, subRouteStr))
        else:
            print(routeStr)
    return


# @jit(nopython=True)
def route_generation(individual, instance):
    route = []
    # vehicleCapacity = instance['vehicle_capacity']
    vehicleCapacity = 4
    # Initialize a sub-route
    subRoute = []
    vehicleLoad = 0
    for customerID in individual[:]: # [[40, 41, 26, 27], [64, 50, 65, 51], [60, 61, 66, 67], [54, 55, 2, 3, 46, 47], [10, 11, 52, 53]]
        # demand = instance['customer_%d' % customerID]['demand']
        demand = int(instance['%d' % customerID]['demand'])
        if customerID % 2 == 1:
            demand = 0
        updatedVehicleLoad = vehicleLoad + demand
        if (updatedVehicleLoad <= vehicleCapacity):
            if customerID % 2 == 1:
                if (customerID-1) in subRoute:
                    # Add to current sub-route
                    subRoute.append(customerID)
                    updatedVehicleLoad -= demand
                else:
                    individual.append(customerID)
            else:
                subRoute.append(customerID)
            vehicleLoad = updatedVehicleLoad
        else:
            for passengers in subRoute[:]:
                if passengers % 2 == 0:
                    if (passengers + 1) not in subRoute:
                        subRoute.append(passengers + 1)
                        # list(filter((customerID + 1).__ne__, individual))
                        # individual.remove(customerID + 1)
                        try:
                            individual.remove(passengers + 1)
                        except:
                            pass

            # Save current sub-route
            route.append(subRoute)
            # Initialize a new sub-route and add to it
            flag = 0
            for list in route:
                if customerID in list:
                    flag = 1
            if flag == 0:
                subRoute = [customerID]
            vehicleLoad = demand

    if subRoute != []:
        for passengers in subRoute[:]:
                if passengers % 2 == 0:
                    if (passengers + 1) not in subRoute:
                        subRoute.append(passengers + 1)
        # Save current sub-route before return if not empty
        route.append(subRoute)
    return route


## sum(w + dr)
# @profile
def eval_GA_1(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    fitness = 0
    for subRoute in route[:]:
        ideal_distance = 0
        distance = 0
        wait = 0
        for i, customerID in enumerate(subRoute, 0):
            if customerID % 2 == 0:
                actual_route = []
                for j, x in enumerate(subRoute[i+1:]):
                    if x != customerID + 1:
                        actual_route.append(x)
                    else:
                        actual_route.append(subRoute[i+j+1])
                        break
                current_psg = customerID
                for next_in_route in actual_route:
                    distance += instance['distance_matrix'][current_psg][next_in_route]
                    current_psg = next_in_route
                ideal_distance += instance['distance_matrix'][customerID][customerID + 1]
                if i != 0:
                    wait_list = subRoute[:i+1]
                    for j, x in enumerate(wait_list[:-1]):
                        wait += instance['distance_matrix'][x][wait_list[j+1]]
        detour = distance - ideal_distance
        total_cost = wait * waitCost + detour * detourCost
        fitness += total_cost #** 2
    return fitness,


## max(w + dr)
def eval_GA_2(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    max_cost = 0
    for subRoute in route[:]:
        ideal_distance = 0
        distance = 0
        wait = 0
        for i, customerID in enumerate(subRoute, 0):
            if customerID % 2 == 0:
                actual_route = []
                for j, x in enumerate(subRoute[i+1:]):
                    if x != customerID + 1:
                        actual_route.append(x)
                    else:
                        actual_route.append(subRoute[i+j+1])
                        break
                current_psg = customerID
                for next_in_route in actual_route:
                    distance += instance['distance_matrix'][current_psg][next_in_route]
                    current_psg = next_in_route
                ideal_distance += instance['distance_matrix'][customerID][customerID + 1]

                if i != 0:
                    wait_list = subRoute[:i+1]
                    for j, x in enumerate(wait_list[:-1]):
                        wait += instance['distance_matrix'][x][wait_list[j+1]]
        detour = distance - ideal_distance
        total_cost = wait * waitCost + detour * detourCost
        if total_cost > max_cost:
            max_cost = total_cost
    fitness = max_cost
    return fitness,


## sum(w)
def eval_GA_3(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    fitness = 0
    for subRoute in route[:]:
        ideal_distance = 0
        distance = 0
        wait = 0
        for i, customerID in enumerate(subRoute, 0):
            if customerID % 2 == 0:
                actual_route = []
                for j, x in enumerate(subRoute[i+1:]):
                    if x != customerID + 1:
                        actual_route.append(x)
                    else:
                        actual_route.append(subRoute[i+j+1])
                        break
                current_psg = customerID
                for next_in_route in actual_route:
                    distance += instance['distance_matrix'][current_psg][next_in_route]
                    current_psg = next_in_route
                ideal_distance += instance['distance_matrix'][customerID][customerID + 1]
                if i != 0:
                    wait_list = subRoute[:i+1]
                    for j, x in enumerate(wait_list[:-1]):
                        wait += instance['distance_matrix'][x][wait_list[j+1]]
        # detour = distance - ideal_distance
        total_cost = wait * waitCost #+ detour * detourCost
        fitness += total_cost #** 2
    return fitness,


# max(w)
def eval_GA_4(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    max_cost = 0
    for subRoute in route[:]:
        ideal_distance = 0
        distance = 0
        wait = 0
        for i, customerID in enumerate(subRoute, 0):
            if customerID % 2 == 0:
                actual_route = []
                for j, x in enumerate(subRoute[i+1:]):
                    if x != customerID + 1:
                        actual_route.append(x)
                    else:
                        actual_route.append(subRoute[i+j+1])
                        break
                current_psg = customerID
                for next_in_route in actual_route:
                    distance += instance['distance_matrix'][current_psg][next_in_route]
                    current_psg = next_in_route
                ideal_distance += instance['distance_matrix'][customerID][customerID + 1]

                if i != 0:
                    wait_list = subRoute[:i+1]
                    for j, x in enumerate(wait_list[:-1]):
                        wait += instance['distance_matrix'][x][wait_list[j+1]]
        # detour = distance - ideal_distance
        total_cost = wait * waitCost #+ detour * detourCost
        if total_cost > max_cost:
            max_cost = total_cost
    fitness = max_cost
    return fitness,


## sum(dr)
def eval_GA_5(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    fitness = 0
    for subRoute in route[:]:
        ideal_distance = 0
        distance = 0
        wait = 0
        for i, customerID in enumerate(subRoute, 0):
            if customerID % 2 == 0:
                actual_route = []
                for j, x in enumerate(subRoute[i+1:]):
                    if x != customerID + 1:
                        actual_route.append(x)
                    else:
                        actual_route.append(subRoute[i+j+1])
                        break
                current_psg = customerID
                for next_in_route in actual_route:
                    distance += instance['distance_matrix'][current_psg][next_in_route]
                    current_psg = next_in_route
                ideal_distance += instance['distance_matrix'][customerID][customerID + 1]
                if i != 0:
                    wait_list = subRoute[:i+1]
                    for j, x in enumerate(wait_list[:-1]):
                        wait += instance['distance_matrix'][x][wait_list[j+1]]
        detour = distance - ideal_distance
        total_cost = detour * detourCost
        fitness += total_cost #** 2
    return fitness,


# max(dr)
def eval_GA_6(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    max_cost = 0
    for subRoute in route[:]:
        ideal_distance = 0
        distance = 0
        wait = 0
        for i, customerID in enumerate(subRoute, 0):
            if customerID % 2 == 0:
                actual_route = []
                for j, x in enumerate(subRoute[i+1:]):
                    if x != customerID + 1:
                        actual_route.append(x)
                    else:
                        actual_route.append(subRoute[i+j+1])
                        break
                current_psg = customerID
                for next_in_route in actual_route:
                    distance += instance['distance_matrix'][current_psg][next_in_route]
                    current_psg = next_in_route
                ideal_distance += instance['distance_matrix'][customerID][customerID + 1]

                if i != 0:
                    wait_list = subRoute[:i+1]
                    for j, x in enumerate(wait_list[:-1]):
                        wait += instance['distance_matrix'][x][wait_list[j+1]]
        detour = distance - ideal_distance
        total_cost = detour * detourCost
        if total_cost > max_cost:
            max_cost = total_cost
    fitness = max_cost
    return fitness,


# sum(dt)
def eval_GA_7(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    total_cost = 0
    # ideal_distance = 0
    for subRoute in route:
        subRoute_distance = 0
        # wait = 0
        for i, customerID in enumerate(subRoute, 0):
            try:
                next_customerID = subRoute[i+1]
            except:
                pass
            else:
                # Calculate section distance
                distance = instance['distance_matrix'][customerID][next_customerID]
                # Update sub-route distance
                subRoute_distance += distance
        total_cost += subRoute_distance
    fitness =  total_cost #** 2
    return fitness,


## max(dt)
def eval_GA_8(individual, instance, unitCost=1.0, initCost=0, waitCost=1, detourCost=1):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    max_cost = 0
    # ideal_distance = 0

    for subRoute in route:
        subRoute_distance = 0
        # wait = 0
        for i, customerID in enumerate(subRoute, 0):
            try:
                next_customerID = subRoute[i+1]
            except:
                pass
            else:
                # Calculate section distance
                distance = instance['distance_matrix'][customerID][next_customerID]
                # Update sub-route distance
                subRoute_distance += distance
        total_cost = subRoute_distance
        if total_cost > max_cost:
            max_cost = total_cost
    fitness =  max_cost #** 2
    return fitness,


# def eval_GA_9(individual, instance):



def avg_dist(individual, instance):
    route = individual
    if not isinstance(individual[0], list):
        route = route_generation(individual, instance)
    total_cost = 0

    for subRoute in route:
        subRoute_distance = 0
        for i, customerID in enumerate(subRoute, 0):
            try:
                next_customerID = subRoute[i+1]
            except:
                pass
            else:
                distance = instance['distance_matrix'][customerID][next_customerID]
                subRoute_distance += distance
        total_cost += subRoute_distance
    fitness =  total_cost/len(route)
    return fitness



def eval_GA_1_dynamic(available_veh, min_req_id, individual, instance, new_req):
    if min_req_id == 0:
        last_loc = available_veh[individual[0]][-1]
    else:
        chosen_veh_index = min_req_id[individual[0]]
        last_loc = available_veh[chosen_veh_index][-1]
    wait = instance['distance_matrix'][last_loc][new_req]
    fitness = wait ** 2
    return fitness,



def cxPartialyMatched(ind1, ind2):
    """
    Executes a partially matched crossover (PMX) on the input individuals.
    The two individuals are modified in place. This crossover expects
    :term:`sequence` individuals of indices, the result for any other type of
    individuals is unpredictable.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    Moreover, this crossover generates two children by matching
    pairs of values in a certain range of the two parents and swapping the values
    of those indexes. For more details see [Goldberg1985]_.

    This function uses the :func:`~random.randint` function from the python base
    :mod:`random` module.

    .. [Goldberg1985] Goldberg and Lingel, "Alleles, loci, and the traveling
       salesman problem", 1985.
    """

    size = min(len(ind1), len(ind2))
    p1, p2 = [0]*size, [0]*size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i
    # Choose crossover points
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
    return ind1, ind2


def cxOrdered(ind1, ind2):
    """
    Executes an ordered crossover (OX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    Moreover, this crossover generates holes in the input
    individuals. A hole is created when an attribute of an individual is
    between the two crossover points of the other individual. Then it rotates
    the element so that all holes are between the crossover points and fills
    them with the removed elements in order. For more details see
    [Goldberg1989]_.
    This function uses the :func:`~random.sample` function from the python base
    :mod:`random` module.
    .. [Goldberg1989] Goldberg. Genetic algorithms in search,
       optimization and machine learning. Addison Wesley, 1989
    """

    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    # a, b = numpy.random.choice(size, 2).tolist()
    if a > b:
        a, b = b, a

    holes1, holes2 = [True]*size, [True]*size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1 , k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def mutInverseIndexes(individual):
    """
    Inverse the sequence in between two randomly selected points in the individual.
    :param individual: Individual to be mutated.
    :return:  A tuple of one individual.
    """
    start, stop = sorted(random.sample(list(range(len(individual))), 2))
    individual = individual[:start] + individual[stop:start-1:-1] + individual[stop+1:]
    return individual,


def mutShuffleIndexes(individual):
    """
    Shuffle the attributes of the input individual and return the mutant.
    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
    probability of each attribute to be moved. Usually this mutation is applied on
    vector of indices.
    :param individual: Individual to be mutated.
    :returns: A tuple of one individual.
    This function uses the :func:`~random.random` and :func:`~random.randint`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)
    for i in range(size):
        swap_indx = random.randint(0, size - 2)
        # swap_indx = numpy.random.random_integers(0, size-2).item()
        if swap_indx >= i:
            swap_indx += 1
        individual[i], individual[swap_indx] = \
            individual[swap_indx], individual[i]
    return individual,


def GA_VRP(instName, unitCost, initCost, waitCost, detourCost, indSize, popSize,
           cxPb, mutPb, NGen, exportCSV=False, customizeData=False):
    if customizeData:
        jsonDataDir = os.path.join(BASE_DIR,'benchmark', 'json_customize')
    else:
        jsonDataDir = os.path.join(BASE_DIR,'benchmark', 'json')
    jsonFile = os.path.join(jsonDataDir, '%s.json' % instName)
    with open(jsonFile) as f:
        instance = load(f)
    creator.create('FitnessMax', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, indSize + 1), indSize)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register('evaluate', eval_GA_1, instance=instance, unitCost=unitCost,
                     initCost=initCost, waitCost=waitCost, detourCost=detourCost)
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cxPartialyMatched)
    toolbox.register('mutate', mutInverseIndexes)
    # Initialize the population
    pop = toolbox.population(n=popSize)
    # Results holders for exporting results to CSV file
    csvData = []

    print( 'Start of evolution')
    # Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Debug, suppress print()
    # print '  Evaluated %d individuals' % len(pop)
    # Begin the evolution
    for g in range(NGen):
        # Debug, suppress print()
        # print '-- Generation %d --' % g
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxPb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutPb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalidInd = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalidInd)
        for ind, fit in zip(invalidInd, fitnesses):
            ind.fitness.values = fit
        # Debug, suppress print()
        # print '  Evaluated %d individuals' % len(invalidInd)

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        # Debug, suppress print()
        # print '  Min %s' % min(fits)
        # print '  Max %s' % max(fits)
        # print '  Avg %s' % mean
        # print '  Std %s' % std

        # Write benchmark to holders for exporting results to CSV file
        if exportCSV:
            csvRow = {
                'generation': g,
                'evaluated_individuals': len(invalidInd),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
                'avg_cost': 1 / mean,
            }
            csvData.append(csvRow)

    print('-- End of (successful) evolution --')
    bestInd = tools.selBest(pop, 1)[0]
    print('Best individual: %s' % bestInd)
    print('Fitness: %s' % bestInd.fitness.values[0])
    print_route(route_generation(bestInd, instance))
    print('Total cost: %s' % (1 / bestInd.fitness.values[0]))

    if exportCSV:
        csvFilename = '%s_uC%s_iC%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_nG%s.csv' % (instName, unitCost, initCost, waitCost,
                                                                               detourCost, indSize, popSize, cxPb, mutPb, NGen)
        csvPathname = os.path.join(BASE_DIR, 'results', csvFilename)
        print('Write to file: %s' % csvPathname)
        makeDirsForFile(pathname=csvPathname)
        if not exist(pathname=csvPathname, overwrite=True):
            with open(csvPathname, 'w') as f:
                fieldnames = ['generation', 'evaluated_individuals', 'min_fitness', 'max_fitness', 'avg_fitness', 'std_fitness', 'avg_cost']
                writer = csv.DictWriter(f, fieldnames=fieldnames, dialect='excel')
                writer.writeheader()
                for csvRow in csvData:
                    writer.writerow(csvRow)
