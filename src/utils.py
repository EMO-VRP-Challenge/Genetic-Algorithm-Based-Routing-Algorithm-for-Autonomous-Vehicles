# -*- coding: utf-8 -*-
# utils.py
import csv
import glob
import os
import fnmatch
import random
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
# import matplotlib.cm as cm
# import numpy as np
from itertools import cycle

import CRZ
from . import BASE_DIR
import numpy as np
from timeit import default_timer as timer
from collections import OrderedDict
import matplotlib.pylab as pl

# BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def makeDirsForFile(pathname):
    try:
        os.makedirs(os.path.split(pathname)[0])
    except:
        pass


def exist(pathname, overwrite=False, displayInfo=True):
    def __pathType(pathname):
        if os.path.isfile(pathname):
            return 'File'
        if os.path.isdir(pathname):
            return 'Directory'
        if os.path.islink(pathname):
            return 'Symbolic Link'
        if os.path.ismount(pathname):
            return 'Mount Point'
        return 'Path'
    if os.path.exists(pathname):
        if overwrite:
            if displayInfo:
                print(('%s: %s exists. Overwrite.' % (__pathType(pathname), pathname)))
            os.remove(pathname)
            return False
        else:
            if displayInfo:
                print(('%s: %s exists.' % (__pathType(pathname), pathname)))

            return True
    else:
        if displayInfo:
            print(('%s: %s does not exist.' % (__pathType(pathname), pathname)))
        return False


def text_to_json(filedir, customize='2'):
    # def __distance(customer1, customer2):
    #     return ((customer1['coordinates']['x'] - customer2['coordinates']['x'])**2 +
    #             (customer1['coordinates']['y'] - customer2['coordinates']['y'])**2)**0.5

    def __distance(customer1, customer2):
        return ((customer1['x'] - customer2['x'])**2 +
                (customer1['y'] - customer2['y'])**2)**0.5
    if customize == '1':
        textDataDir = os.path.join(BASE_DIR, 'benchmark', 'text_customize')
        jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'json_customize')
    elif customize == '0':
        textDataDir = os.path.join(BASE_DIR, 'benchmark', 'text')
        jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'json')
    else:
        # textDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split', 'Cluster') # , 'Cluster'
        # textDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split', 'Random')
        # textDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ', CRZ.ty)
        # jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split_json', 'C')
        # jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split_json', 'R')
        jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_json', CRZ.ty)
        os.makedirs(jsonDataDir, exist_ok=True)


    # for filedir in glob.iglob(os.path.join(textDataDir, '*.csv')):
    start_time = timer()
    name = os.path.splitext(os.path.basename(filedir))[0] + '.json'
    # print(name)
    df = pd.read_csv(filedir)
    dist_list = []
    for i in range(df.shape[0]): # df.shape[0]
        sub_dist_list = []
        for j in range(df.shape[0]):
            dist = np.sqrt(np.square(df.iloc[i][1] - df.iloc[j][1]) + np.square(df.iloc[i][2] - df.iloc[j][2]))
            sub_dist_list.append(dist)
        dist_list.append(sub_dist_list)
        print(str(i) + '---' + name)
    reader = csv.DictReader(open(filedir))
    data = {}
    for row in reader:
        key = row.pop('ID')
        data[key] = row

    data['distance_matrix'] = dist_list

    jsonPath = os.path.join(jsonDataDir, name)
    with open(jsonPath, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))
    computing_time = (timer() - start_time)/60
    print('Computing Time: %s min' % computing_time)



def result_analyze():
    plt.style.use('ggplot')
    resultsDataDir = os.path.join(BASE_DIR, 'results', 'analysis')
    for csvFile in [os.path.join(resultsDataDir, csvFilename) for csvFilename in fnmatch.filter(os.listdir(resultsDataDir), '*csv')]:
        df = pd.read_csv(csvFile)
        # Delete the last 2 columns (std_fitness, avg_cost)
        if len(df.columns) == 6:
            df.drop(df.columns[[5]], axis=1, inplace=True)
        else:
            df.drop(df.columns[[5, 6]], axis=1, inplace=True)
        # Delete the first 2 columns (generation, evaluated_individuals)
        df.drop(df.columns[[0, 1, 3, 4]], axis=1, inplace=True)

        plt.figure()
        ax = df.plot(figsize=(40, 30)) # 20, 10
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')

        # plt.show()
        # Save to folder
        filename = os.path.splitext(csvFile)[0]
        plt.savefig(filename + '.png')


# def visualization(threshold, no, instanceName, instance, route, crossover, mutation, select, waitCost,
#                   detourCost, indSize, popSize=None, cxPb=None, mutPb=None, NGen=None):
def visualization(nfolder, ty, algName, instanceName, instance, route, crossover, mutation, select, waitCost,
                  detourCost, indSize, popSize=None, cxPb=None, mutPb=None, NGen=None):
    # Takes in the instance and a route list and plots the routes.

    # df = pd.DataFrame()
    # for customer in instance:
    #     if customer.startswith("customer"):
    #         row = [[str(customer), instance[customer]["coordinates"]["x"], instance[customer]["coordinates"]["y"]]]
    #         df = df.append(row, ignore_index=True)
        # else:
        #     next

    # for key, value in instance.items():

    row = []
    i = 0
    # for i in range(2000):
    for customer in instance:
        try:
            # row.append([i, int(instance.get("i")["x"]), int(instance.get("i")["y"])])
            row.append([int(customer), int(instance[customer]["x"]), int(instance[customer]["y"])])
        except:
            pass
            # print(i)
        i += 1
    df = pd.DataFrame(row)
    # df = df[:200]
    df.columns = ['customer_number', 'x_pos', 'y_pos']
    df.set_index('customer_number', inplace=True)
    plt.style.use('ggplot')
    # plotDataDir = os.path.join(BASE_DIR, 'plot', 'TEST', foldername, algName)
    # plotDataDir = os.path.join(BASE_DIR, 'plot', '1000req', nfolder, foldername, algName)
    plotDataDir = os.path.join(BASE_DIR, 'plot', '100req', ty)
    # plotDataDir = os.path.join(BASE_DIR, 'plot', 'CRZ_1dmd', foldername, algName)
    # plotDataDir = os.path.join(BASE_DIR, 'plot', 'CRZ_split', foldername, algName)

    # try:
    #     os.makedirs(plotDataDir)
    # except:
    #     pass

    # Plot the scatter plot of customers
    # plt.figure()
    ax = df.plot.scatter(x='x_pos', y='y_pos', color='red', s=1, figsize=(10, 10))
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 1000))
    # Label the axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # define different colors for different routes
    # colors = cycle(["b", "r",  "g", "purple", "orange", "grey", "black", "pink", "y", "brown"])
    # colors = iter(plt.cm.rainbow(np.linspace(0,1,1000)))
    colors = iter(pl.cm.jet(np.linspace(0,1,len(route)))) # 1000
    # Plot the connections of routes:

    # colors = cm.rainbow(np.linspace(0, 1, len(route)))

    for subRoutes in route:
        color=next(colors)
        first_passenger = subRoutes[0]

        # ax.text(float( df.loc[df['customer_number'] == "customer_%d" % first_passenger] ['x_pos'] ),
        #         float( df.loc[df['customer_number'] == "customer_%d" % first_passenger] ['y_pos'] ), str(first_passenger), color = 'red') #
        # for customer, next_customer in zip(subRoutes[0::], subRoutes[1::]):
        #     i, j = list(zip([float( df.loc[df['customer_number'] == "customer_%d" % customer] ['x_pos'] ),
        #                       float( df.loc[df['customer_number'] == "customer_%d" % customer] ['y_pos'] )],
        #                      [float( df.loc[df['customer_number'] == "customer_%d" % next_customer] ['x_pos'] ),
        #                       float( df.loc[df['customer_number'] == "customer_%d" % next_customer] ['y_pos'] )]))

        # ax.text(float( df.loc[df['customer_number'] == first_passenger] ['x_pos'] ),
        #         float( df.loc[df['customer_number'] == first_passenger] ['y_pos'] ), str(first_passenger), color = 'red', fontsize=10) #
        ax.text(float( df.loc[first_passenger] ['x_pos'] ),
                float( df.loc[first_passenger] ['y_pos'] ), str(first_passenger), color = 'red', fontsize=8)
        for customer, next_customer in zip(subRoutes[0::], subRoutes[1::]):
            # i, j = list(zip([float( df.loc[df['customer_number'] == customer] ['x_pos'] ),
            #                   float( df.loc[df['customer_number'] == customer] ['y_pos'] )],
            #                  [float( df.loc[df['customer_number'] == next_customer] ['x_pos'] ),
            #                   float( df.loc[df['customer_number'] == next_customer] ['y_pos'] )]))
            # plt.plot(i, j, color) # colormap='rainbow'

            x_list = [df.loc[customer]['x_pos'], df.loc[next_customer]['x_pos']]
            y_list = [df.loc[customer]['y_pos'], df.loc[next_customer]['y_pos']]
            # plot routes
            plt.plot(x_list, y_list, color=color) # , alpha=0.5

            # Annotate the customer numbers
            # ax.text(i[0], j[0], str(customer))
            ax.text(x_list[1], y_list[1], str(next_customer), fontsize=8)
    if 'Z' in instanceName:
        circle1 = plt.Circle((500, 500), radius=100, color='grey', fill=False, linewidth=2)
        circle2 = plt.Circle((500, 500), radius=300, color='grey', fill=False, linewidth=2)
        circle3 = plt.Circle((500, 500), radius=500, color='grey', fill=False, linewidth=2)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(circle3)

    # Save to folder
    currentTime = time.strftime("_%m-%d_%H-%M-%S")
    try:
        # parameters = '_cx%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_f%s_%s' % (crossover, mutation, select, waitCost,
        #                                                                          detourCost, indSize, popSize, cxPb, mutPb, threshold, no)
        # parameters = '_cro%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_nG%s' % (crossover, mutation, select, waitCost,
        #                                                                               detourCost, indSize, popSize, cxPb, mutPb, NGen)
        parameters = '_%s_indS%s_popS%s_nG%s' % (algName, indSize, popSize, NGen)
    except:
        parameters = currentTime
    file_name = parameters + '.png'
    file_location = os.path.join(plotDataDir, instanceName+file_name)
    makeDirsForFile(pathname=file_location)
    # if exist(pathname=file_location, overwrite=False):
    #     parameters = '_%s_indS%s_popS%s_nG%s_1' % (algName, indSize, popSize, NGen)
    #     file_name = parameters + '.png'
    #     file_location = os.path.join(plotDataDir, instanceName+file_name)
    fig = ax.get_figure()
    fig.savefig(file_location)
    plt.close('all')

