
#  -*- coding: utf-8 -*-
# CRZ_split.py
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
import cProfile, pstats, io
from numba import jit
# from scoop import futures

# sys.setrecursionlimit(10000)
# Global constant for individual size
# Check before running
# IND_SIZE = 8
# IND_SIZE = 2000
# threshold = 0.001


def profile(fnc):

    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner


# @jit(nopython=True)
@profile
def GA(filedir):

    """
    In this GA implementation, the individuals are valid throughout the GA process, they are made valid
    right after the population are generated and right after each crossover and mutation operation.
    (in the check_validity function)
    """
    instName = os.path.splitext(os.path.basename(filedir))[0]
    with open(filedir) as in_f:
        instance = load(in_f)
    crossover='Ord'
    # mutation='Inv'
    mutation='Shu'
    # select='Rou'
    select='Tour'
    unitCost = 1.0
    waitCost = 1.0
    detourCost = 1.0
    initCost = 0.0
    mutPb = 0.3
    cxPb = 0.7
    popSize = 100
    NGen = 100
    exportCSV = True
    ty = 'C'
    nfolder = '15'
    aggregate_list = []
    for algNo in range(1, 9):
        alg = []
        obj1 = []
        obj2 = []
        obj3 = []
        obj4 = []
        obj5 = []
        obj6 = []
        obj7 = []
        obj8 = []
        # start_time = timer()
        # folder = os.path.join(BASE_DIR, 'benchmark')
        # sfolder = os.path.join(folder, 'CRZ_json', 'test')
        # for filedir in glob.iglob(os.path.join(sfolder, '*.json')):
        start_time = timer()
        # instName = os.path.splitext(os.path.basename(filedir))[0]

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
        # fits = [ind.fitness.values[0] for ind in pop]

        # g = 0

        # Begin the evolution
        for g in range(NGen):
        # while min(fits) > threshold:
            print('-- Generation %d --' % g)
            # print(f'Process {os.getpid()} working.')
            # proc_name = multiprocessing.current_process().name
            # print(f'Current process: {proc_name}.')
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
            # fits = [ind.fitness.values[0] for ind in pop]
            # length = len(pop)
            # mean = sum(fits) / length
            # sum2 = sum(x*x for x in fits)
            # std = abs(sum2 / length - mean**2)**0.5
        print('-- End of evolution --')
        computing_time = (timer() - start_time)
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
        if algNo == 1:
            final_fit = core.eval_GA_1(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 2:
            final_fit = core.eval_GA_2(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 3:
            final_fit = core.eval_GA_3(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 4:
            final_fit = core.eval_GA_4(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 5:
            final_fit = core.eval_GA_5(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 6:
            final_fit = core.eval_GA_6(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 7:
            final_fit = core.eval_GA_7(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        elif algNo == 8:
            final_fit = core.eval_GA_8(best_route, instance=instance, unitCost=unitCost, initCost=initCost,
                                  waitCost=waitCost, detourCost=detourCost)[0]
        # max_req = len(max(best_route))/2
        # print(f'Max psg per veh: {max_req}')
        print('Best route: %s' % best_route)
        # sharing = []
        current_psg = []
        for veh in best_route:
            num_psg = 0
            for psg in veh:
                if psg % 2 == 0:
                    num_psg += 1
                else:
                    num_psg -= 1
                # if num_psg > 1:
                current_psg.append(num_psg)
        avg_sharing = numpy.mean(current_psg)

        avg_cross = -1
        if ty == 'C':
            crossing = []
            for veh in best_route:
                cross = 0
                if -1 < veh[0] < 50:
                    for point in veh[1:]:
                        if point > 49:
                            cross += 1
                if 49 < veh[0] < 100:
                    for point in veh[1:]:
                        if point > 99 or point < 50:
                            cross += 1
                if 99 < veh[0] < 150:
                    for point in veh[1:]:
                        if point > 149 or point < 100:
                            cross += 1
                if 149 < veh[0] < 200:
                    for point in veh[1:]:
                        if point < 150:
                            cross += 1
                crossing.append(cross)
            avg_cross = numpy.mean(crossing)

        if exportCSV:
            csvRow = {
                'avg_cross': avg_cross,
                'num_veh': len(best_route),
                'avg_req': avg_req,
                # 'avg_psg': avg_psg,
                # 'max_req': max_req,
                'avg_sharing': avg_sharing,
                'avg_dist': core.avg_dist(best_route, instance),
                'computing_time(s)': computing_time,
                'route': best_route,
                'min_fitness': final_fit
            }
            csvData.append(csvRow)
            csvFilename = '%s_alg%s_indS%s_popS%s_nG%s.csv' % (instName, algNo, IND_SIZE, popSize, NGen)

            # csvPathname = os.path.join(BASE_DIR, 'results', 'CRZ', nfolder, ty, algName, csvFilename)
            csvPathname = os.path.join(BASE_DIR, 'results', '100req', ty, csvFilename)
            utils.makeDirsForFile(pathname=csvPathname)
            if utils.exist(pathname=csvPathname, overwrite=False):
                # csvFilename = '%s_alg%s_cro%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_nG%s.csv' % (instName, algNo, crossover, mutation, select, waitCost,
                #                                                                              detourCost, IND_SIZE, popSize, cxPb, mutPb, NGen)
                csvFilename = '%s_alg%s_indS%s_popS%s_nG%s_1.csv' % (instName, algNo, IND_SIZE, popSize, NGen)
                csvPathname = os.path.join(BASE_DIR, 'results', '100req', ty, csvFilename)
            with open(csvPathname, 'w') as f:
                    # fieldnames = ['min_fitness', 'num_veh', 'avg_req', 'avg_dist', 'computing_time(s)']
                    fieldnames = ['avg_cross', 'num_veh', 'avg_req', 'avg_sharing', 'avg_dist', 'computing_time(s)', 'route', 'min_fitness']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, dialect='excel')
                    writer.writeheader()
                    for csvRow in csvData:
                        writer.writerow(csvRow)
            print('Write to file: %s' % csvPathname)
                    # writer = csv.writer(f)
                    # writer.writerow(row)
        # best_route = core.route_generation(bestInd, instance)
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
        utils.visualization(nfolder, ty, algName, instName, instance, best_route, crossover, mutation, select, waitCost,
                            detourCost, IND_SIZE, popSize, cxPb, mutPb, NGen)
        # print('Computing Time: %s min' % computing_time)
        print('Computing Time: %s s' % computing_time)
        alg.append(numpy.mean(obj1))
        alg.append(numpy.mean(obj2))
        alg.append(numpy.mean(obj3))
        alg.append(numpy.mean(obj4))
        alg.append(numpy.mean(obj5))
        alg.append(numpy.mean(obj6))
        alg.append(numpy.mean(obj7))
        alg.append(numpy.mean(obj8))
        aggregate_list.append(alg)
    df_objs = pd.DataFrame(aggregate_list).T
    df_objs.replace(0.0, 500.0, inplace=True)
    df_objs.index = ['obj1', 'obj2', 'obj3', 'obj4', 'obj5', 'obj6', 'obj7', 'obj8']
    # df_objs.columns = ['obj1', 'obj7']
    df_objs.columns = ['alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'alg6', 'alg7', 'alg8']
    # df_objs.index = ['alg2']
    plot = df_objs.plot.line(colormap='rainbow', figsize=(10,10), linewidth=1) # logy=True,
    plot.set(xlabel='objective', ylabel='fitness')
    fig = plot.get_figure()
    # fig_path = os.path.join(BASE_DIR, 'results', 'CRZ', ty)
    fig_path = os.path.join(BASE_DIR, 'results', '100req')
    fig.savefig(fig_path + '/' + ty + '.png')
    # fig.savefig(fig_path + '/'  + 'C.png')
    res_name = ty + '.csv'
    # res_name = 'C.csv'
    pathout = os.path.join(fig_path, res_name)
    df_objs.to_csv(pathout) # , index=False
    return best_route


if __name__ == '__main__':
    start_time_0 = timer()
    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool()
    IND_SIZE = 200
    # Create Fitness and Individual Classes
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register('indexes', random.sample, range(0, IND_SIZE), IND_SIZE)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # toolbox.register('map', pool.map)
    # toolbox.register("map", futures.map)
    # main()
    random.seed()
    # pool.map(GA, args=(crossover, mutation, select, unitCost, initCost, waitCost, detourCost, popSize, NGen,))
    # GA()
    # p = multiprocessing.Process(target=GA, args=(crossover, mutation, select, unitCost, initCost, waitCost, detourCost, popSize, NGen,))
    folder = os.path.join(BASE_DIR, 'benchmark')
    sfolder = os.path.join(folder, 'CRZ_json')
    for filedir in glob.iglob(os.path.join(sfolder, '*.json')):
        GA(filedir)
    total_computing_time = round((timer() - start_time_0)/60, 2)
    print(f'TOTAL_TIME: {total_computing_time} min.')
    # p.start()
    # p.join()
    os.system('afplay /System/Library/Sounds/Glass.aiff')
    # pool.close()

