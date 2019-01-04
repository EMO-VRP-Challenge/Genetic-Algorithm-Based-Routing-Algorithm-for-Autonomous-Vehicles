#  -*- coding: utf-8 -*-
# A-n16-k4.py
import glob
import math
import os
import random
import sys
import numpy
from json import load
import csv
from deap import base, creator, tools
from timeit import default_timer as timer
import multiprocessing
from src import core, utils, BASE_DIR
from itertools import chain
import functools
import pandas as pd
# from scoop import futures

# sys.setrecursionlimit(10000)
# Global constant for individual size
# Check before running
# IND_SIZE = 8
# IND_SIZE = 2000
threshold = 0.001

# # Create Fitness and Individual Classes
# creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
# creator.create('Individual', list, fitness=creator.FitnessMin)
#
# """
#     create("Foo", list, bar=dict, spam=1)
#
#     This above line is exactly the same as defining in the :mod:`creator`
#     module something like the following:
#
#     class Foo(list):
#         spam = 1
#         def __init__(self):
#             self.bar = dict()
#
# """
#
# toolbox = base.Toolbox()
# # Attribute generator
# toolbox.register('indexes', random.sample, range(0, IND_SIZE), IND_SIZE)
# # Structure initializers
# toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
# toolbox.register('population', tools.initRepeat, list, toolbox.individual)
#
# """
#     The following code block is an example of how the toolbox is used:
#
#         >>> def func(a, b, c=3):
#         ...     print a, b, c
#         ...
#         >>> tools = Toolbox()
#         >>> tools.register("myFunc", func, 2, c=4)
#         >>> tools.myFunc(3)
#         2 3 4
#
# """

def GA(crossover, mutation, select, unitCost, initCost, waitCost, detourCost,
       popSize, NGen, exportCSV=False, customizeData='0'):
            # indSize, popSize, NGen, exportCSV=False, customizeData='0'):

    """
    In this GA implementation, the individuals are valid throughout the GA process, they are made valid
    right after the population are generated and right after each crossover and mutation operation.
    (in the check_validity function)
    """

    # if customizeData == '1':
    #     jsonDataDir = os.path.join('benchmark', 'json_customize')
    #     jsonFile = os.path.join(jsonDataDir, '%s.json' % instName)
    # elif customizeData == '0':
    #     jsonDataDir = os.path.join('benchmark', 'json')
    #     jsonFile = os.path.join(jsonDataDir, '%s.json' % instName)
    # else:
    #     jsonFile = os.path.join('benchmark', 'Solomon_json')

    ty = 'R'
    # name = str(IND_SIZE) + 'req'
    aggregate_list = []
    for algNo in range(1, 9):
    # for algNo in range(1, 2):
        alg = []
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        obj5 = []
        obj6 = []
        obj7 = []
        obj8 = []
        start_time = timer()
        folder = os.path.join(BASE_DIR, 'benchmark')
        # sfolder = os.path.join(folder, 'test_json', name)
        # sfolder = os.path.join(folder, 'CRZ_json', ty)
        sfolder = os.path.join(folder, 'CRZ_split_json', ty)
        for filedir in glob.iglob(os.path.join(sfolder, '*.json')):
            instName = os.path.splitext(os.path.basename(filedir))[0]
            # for mutPb in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            #     for cxPb in [0.6, 0.7, 0.8]:
            for mutPb in [0.3]: #0.1, , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
                for cxPb in [0.7]: #,1600, 1700,1800, 1900,2000 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, , 0.8, 0.9
                    # for popSize in [2000]: # 100, 500, 1000, 1500
                    # cxPb = 0.7
                    # mutPb = 0.3
                    for popSize, NGen in zip([30], [50]): # 1500
                        # for no in range(10):
                        with open(filedir) as in_f:
                            instance = load(in_f)

                        counter = 0
                        for key, value in instance.items():
                            if key != 'distance_matrix':
                                counter += 1
                        IND_SIZE = counter
                        # Create Fitness and Individual Classes
                        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
                        creator.create('Individual', list, fitness=creator.FitnessMin)
                        toolbox = base.Toolbox()
                        # Attribute generator
                        toolbox.register('indexes', random.sample, range(0, IND_SIZE), IND_SIZE)
                        # Structure initializers
                        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
                        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

                        pool = multiprocessing.Pool()
                        toolbox.register('map', pool.map)

                        if algNo == 1:
                            # Operator registering
                            toolbox.register('evaluate', core.eval_GA_1, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 2:
                            toolbox.register('evaluate', core.eval_GA_2, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 3:
                            toolbox.register('evaluate', core.eval_GA_3, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 4:
                            toolbox.register('evaluate', core.eval_GA_4, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 5:
                            toolbox.register('evaluate', core.eval_GA_5, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 6:
                            toolbox.register('evaluate', core.eval_GA_6, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 7:
                            toolbox.register('evaluate', core.eval_GA_7, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        elif algNo == 8:
                            toolbox.register('evaluate', core.eval_GA_8, instance=instance, unitCost=unitCost, initCost=initCost,
                                             waitCost=waitCost, detourCost=detourCost)
                        if select == 'Rou':
                            toolbox.register('select', tools.selRoulette)
                        elif select == 'Tour':
                            toolbox.register('select', tools.selTournament,  tournsize=1)
                        if crossover == 'PM':
                            toolbox.register('mate', core.cxPartialyMatched)
                        elif crossover == 'Ord':
                            toolbox.register('mate', core.cxOrdered)
                        if mutation == 'Inv':
                            toolbox.register('mutate', core.mutInverseIndexes)
                        elif mutation == 'Shu':
                            toolbox.register('mutate', core.mutShuffleIndexes)

                        def check_validity():
                            def decorator(func):
                                @functools.wraps(func)
                                def wrapper(*args, **kargs):
                                    offspring = func(*args, **kargs)
                                    if isinstance(offspring, list):
                                        for individual in offspring:
                                            ind = core.route_generation(individual, instance)
                                            individual[:] = list(chain.from_iterable(ind))
                                        return offspring
                                    else:
                                        offspring = list(offspring)
                                        for individual in offspring:
                                            ind = core.route_generation(individual, instance)
                                            individual[:] = list(chain.from_iterable(ind))
                                        return tuple(offspring)
                                return wrapper
                            return decorator

                        toolbox.decorate("population", check_validity())
                        toolbox.decorate("mate", check_validity())
                        toolbox.decorate("mutate", check_validity())

                        pop = toolbox.population(n=popSize)

                        # Results holders for exporting results to CSV file
                        csvData = []
                        # print('Start of evolution')
                        # Evaluate the entire population
                        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
                        for ind, fit in zip(pop, fitnesses):
                            ind.fitness.values = fit
                        # Debug, suppress print()
                        # print('  Evaluated %d individuals' % len(pop))

                        # Extracting all the fitnesses of
                        fits = [ind.fitness.values[0] for ind in pop]

                        # g = 0

                        # Begin the evolution
                        for g in range(NGen):
                        # while min(fits) > threshold:
                            print('-- Generation %d --' % g)
                            # g += 1

                            # Select the next generation individuals
                            # Select elite - the best offspring, keep this past crossover/mutate
                            elite = tools.selBest(pop, 1)

                            # Keep top 10% of all offspring
                            # use tournament method selects the rest 90% of the offsprings
                            offspring = tools.selBest(pop, int(numpy.ceil(len(pop)*0.1)))
                            offspring_tournament = toolbox.select(pop, int(numpy.floor(len(pop)*0.9))-1)
                            offspring.extend(offspring_tournament)

                            # offspring = toolbox.select(pop, len(pop))

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
                            # print ('Evaluated %d individuals' % len(invalidInd))

                            offspring.extend(elite)
                            # The population is entirely replaced by the offspring
                            pop[:] = offspring

                            # Gather all the fitnesses in one list and print the stats
                            fits = [ind.fitness.values[0] for ind in pop]
                            length = len(pop)
                            mean = sum(fits) / length
                            sum2 = sum(x*x for x in fits)
                            std = abs(sum2 / length - mean**2)**0.5

                            # Debug, suppress print()

                            # print('  Min fitness: %s' % min(fits))

                            # print('  Max %s' % max(fits))
                            # print('  Avg %s' % mean)
                            # print('  Std %s' % std)

                            # Write benchmark to holders for exporting results to CSV file
                            # if exportCSV:
                            #     csvRow = {
                            #         'generation': g,
                            #         # 'evaluated_individuals': len(invalidInd),
                            #         # 'c'+str(cxPb)+'m'+str(mutPb): min(fits),
                            #         # str(instName)+'p'+str(popSize)+'c'+str(cxPb)+'m'+str(mutPb): min(fits),
                            #         'min_fitness': min(fits),
                            #         'num_veh': len(),
                            #         # 'max_fitness': max(fits),
                            #         # 'avg_fitness': mean,
                            #         # 'std_fitness': std,
                            #         # 'avg_cost': 1 / mean,
                            #     }
                            #     csvData.append(csvRow)

                        print('-- End of evolution --')

                        computing_time = (timer() - start_time)/60
                        # row = ['computing time (min) ', computing_time]
                        bestInd = tools.selBest(pop, 1)[0]

                        best_route = core.route_generation(bestInd, instance)

                        # print('Best individual: %s' % bestInd)
                        # f"Best individual: {bestInd}"
                        # print('Fitness: %s' % bestInd.fitness.values[0])

                        # core.print_route(core.route_generation(bestInd, instance))
                        # print('Total cost: %s' % (math.sqrt(bestInd.fitness.values[0])))
                        algName = 'alg' + str(algNo)
                        no_req = []
                        for veh in best_route:
                            no_req.append(len(veh)/2)
                        avg_req = numpy.mean(no_req)

                        final_fit = core.eval_GA_2(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0]

                        if exportCSV:
                            csvRow = {
                                'min_fitness': final_fit,
                                'num_veh': len(best_route),
                                'avg_req': avg_req,
                                'avg_dist': core.avg_dist(best_route, instance),
                                'computing_time(min)': computing_time
                            }
                            csvData.append(csvRow)

                        # if exportCSV:
                            csvFilename = '%s_cro%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_nG%s.csv' % (instName, crossover, mutation, select, waitCost,
                                                                                                             detourCost, IND_SIZE, popSize, cxPb, mutPb, NGen)
                            # csvFilename = '%s_cro%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_f%s_%s.csv' % (instName, crossover, mutation, select, waitCost,
                            #                                                                                  detourCost, indSize, popSize, cxPb, mutPb, threshold, no)
                            # subfolder2 = 'c' + str(cxPb)
                            # subfolder1 = 'm' + str(mutPb)
                            # csvPathname = os.path.join('results', 'fit0.0001', subfolder1, subfolder2, csvFilename)
                            # foldername = instName #+ str(setno)

                            # subname = 'f' + str(threshold)
                            # csvPathname = os.path.join(BASE_DIR, 'results', 'test', '10req', 'alg1', subname, csvFilename) #  '1',, 'gen', 'g200',

                            # csvPathname = os.path.join(BASE_DIR, 'results', 'test', name, algName, csvFilename)
                            # csvPathname = os.path.join(BASE_DIR, 'results', 'CRZ', ty, algName, csvFilename)
                            csvPathname = os.path.join(BASE_DIR, 'results', 'CRZ_split', ty, algName, csvFilename)
                            print('Write to file: %s' % csvPathname)
                            utils.makeDirsForFile(pathname=csvPathname)
                            if not utils.exist(pathname=csvPathname, overwrite=True):
                                with open(csvPathname, 'w') as f:
                                    fieldnames = ['min_fitness', 'num_veh', 'avg_req', 'avg_dist', 'computing_time(min)']
                                    # fieldnames = ['generation', str(instName)+'p'+str(popSize)+'c'+str(cxPb)+'m'+str(mutPb), 'max_fitness',
                                    #               'avg_fitness', 'std_fitness', 'avg_cost']
                                    # fieldnames = ['generation', 'evaluated_individuals', 'set2', 'max_fitness',
                                    #               'avg_fitness', 'std_fitness', 'avg_cost']
                                    writer = csv.DictWriter(f, fieldnames=fieldnames, dialect='excel')
                                    writer.writeheader()
                                    for csvRow in csvData:
                                        writer.writerow(csvRow)
                                    # writer = csv.writer(f)
                                    # writer.writerow(row)
                        best_route = core.route_generation(bestInd, instance)

                        obj1.append(core.eval_GA_1(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj2.append(core.eval_GA_2(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj3.append(core.eval_GA_3(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj4.append(core.eval_GA_4(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj5.append(core.eval_GA_5(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj6.append(core.eval_GA_6(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj7.append(core.eval_GA_7(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        obj8.append(core.eval_GA_8(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                                  waitCost=waitCost, detourCost=detourCost)[0])
                        # aggregate_list.append(obj)
                        print('Best route: %s' % best_route)
                        # utils.visualization(threshold, no, instName, instance, best_route, crossover, mutation, select, waitCost,
                        #                     detourCost, indSize, popSize, cxPb, mutPb, NGen)
                        # utils.visualization(name, threshold, algName, instName, instance, best_route, crossover, mutation, select, waitCost,
                        #                     detourCost, indSize, popSize, cxPb, mutPb, NGen)
                        utils.visualization(ty, threshold, algName, instName, instance, best_route, crossover, mutation, select, waitCost,
                                            detourCost, IND_SIZE, popSize, cxPb, mutPb, NGen)
                        print('Computing Time: %s min' % computing_time)
        alg.append(numpy.mean(obj1))
        alg.append(numpy.mean(obj2))
        alg.append(numpy.mean(obj3))
        alg.append(numpy.mean(obj4))
        alg.append(numpy.mean(obj5))
        alg.append(numpy.mean(obj6))
        alg.append(numpy.mean(obj7))
        alg.append(numpy.mean(obj8))
        aggregate_list.append(alg)
    df_objs = pd.DataFrame(aggregate_list)
    df_objs.columns = ['obj1', 'obj2', 'obj3', 'obj4', 'obj5', 'obj6', 'obj7', 'obj8']
    # df_objs.columns = ['obj1', 'obj7']
    df_objs.index = ['alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'alg6', 'alg7', 'alg8']
    # df_objs.index = ['alg2']
    plot = df_objs.plot.line(colormap='rainbow', figsize=(10,10))
    fig = plot.get_figure()
    # fig_path = os.path.join(BASE_DIR, 'results', 'test', name)
    fig_path = os.path.join(BASE_DIR, 'results', 'CRZ_split', ty)
    # fig.savefig(fig_path + '/' + name + '.png')
    fig.savefig(fig_path + '/' + ty + '-0.png')
    # res_name = name + '.csv'
    res_name = ty + '-0.csv'
    pathout = os.path.join(fig_path, res_name)
    df_objs.to_csv(pathout) # , index=False
    pool.close()
    return best_route


def main():
    random.seed() # 64
    # instName = 'A-n32-k5'
    # instName = 'A-n16-k4'
    # instName = 'A-n32-k5_v1'
    # instName = 'R101-s'
    # instName = 'C101'
    # instName = 'Li32_n1200_k11'
    # instName = 'RC101-s'

    unitCost = 1.0
    waitCost = 1.0
    detourCost = 1.0
    initCost = 0.0
    # indSize = IND_SIZE
    popSize = 1000 # 30

    NGen = 250 # '(fit0.0003)' #800

    exportCSV = True
    # exportCSV = False
    # customizeData = True
    customizeData = '2'

    # folder = os.path.join(BASE_DIR, 'benchmark')
    # sfolder = os.path.join(folder, 'Solomon_json')

    # for mutPb in numpy.l-inspace(0.0, 1.0, num=10, endpoint=False):

    # for setno in [13, 14, 15, 16, 17, 18, 19, 20]:
    #     try:
    # for popSize in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]: # 100, 200, 300, 400, 500, 600, 700, 800, 900,
    # for popSize in [1000, 2000, 3000, 4000, 5000]:
    # # for popSize in [1600, 1700, 1800, 1900, 2000]:
    #
    # for filedir in glob.iglob(os.path.join(sfolder, '*.json')):
    #     instName = os.path.splitext(os.path.basename(filedir))[0]
    #     for mutPb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
    #         for cxPb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #,1600, 1700,1800, 1900,2000
    #             start_time = timer()
    GA(
        # file=filedir,
        # instName=instName,
        # crossover='PM',
        crossover='Ord',
        # mutation='Inv',
        mutation='Shu',
        # select='Rou',
        select='Tour',
        unitCost=unitCost,
        initCost=initCost,
        waitCost=waitCost,
        detourCost=detourCost,
        # indSize=indSize,
        popSize=popSize,
        # cxPb=cxPb,
        # mutPb=mutPb,
        NGen=NGen,
        # start_time=start_time,
        exportCSV=exportCSV,
        customizeData=customizeData
        )


    # for filedir in glob.iglob(os.path.join(sfolder, '*.json')):
    #     instName = os.path.splitext(os.path.basename(filedir))[0]
    #     # instName = 'C106'
    #     start_time = timer()
    #     GA(
    #         file=filedir,
    #         instName=instName,
    #         # crossover='PM',
    #         crossover='Ord',
    #         # mutation='Inv',
    #         mutation='Shu',
    #         # select='Rou',
    #         select='Tour',
    #         unitCost=unitCost,
    #         initCost=initCost,
    #         waitCost=waitCost,
    #         detourCost=detourCost,
    #         indSize=indSize,
    #         popSize=popSize,
    #         cxPb=cxPb,
    #         mutPb=mutPb,
    #         NGen=NGen,
    #         start_time=start_time,
    #         exportCSV=exportCSV,
    #         customizeData=customizeData
    #     )

    return

if __name__ == '__main__':
    # pool = multiprocessing.Pool()
    # toolbox.register('map', pool.map)
    # toolbox.register("map", futures.map)
    main()
    # pool.close()
