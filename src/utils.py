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


def text_to_json(customize='0'):
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
        textDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split', 'Zones')
        # jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split_json', 'C')
        # jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split_json', 'R')
        jsonDataDir = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split_json', 'Z')
        os.makedirs(jsonDataDir, exist_ok=True)


    for filedir in glob.iglob(os.path.join(textDataDir, '*.csv')):
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
        # matrix = {'distance_matrix' : dist_list}

        # with open(filedir) as f:
        #     lines = f.readlines()

        # with open(filedir) as f:
            # lines = csv.reader(filedir)
            # for row in csv.DictReader(f):
            #     data.append(row)
        reader = csv.DictReader(open(filedir))
        data = {}
        for row in reader:
            key = row.pop('ID')
            data[key] = row

        data['distance_matrix'] = dist_list
            # lines = f.readlines()
        # rows = []
        # for line in lines:
        #     rows.append(str(line))
        #     # rows.append(line.split(' '))
        # dist_list = []
        # for i in range(1,2001):
        #     sub_dist_list = []
        #     for j in range(1,2001):
        #         sub_dist_list.append(((int(rows[i][1]) - int(rows[j][1]))**2 + (int(rows[i][2]) - int(rows[j][2]))**2)**0.5)
        #         # sub_dist_list.append(((int(lines[i][1]) - int(lines[j][1]))**2 + (int(lines[i][2]) - int(lines[j][2]))**2)**0.5)
        #     dist_list.append(sub_dist_list)
        # data.append(OrderedDict([('distance_matrix', dist_list)]))
        # 'distance_matrix'+ dist_list
        jsonPath = os.path.join(jsonDataDir, name)
        # os.makedirs(jsonPath, exist_ok=True)
        # if not os.path.exists(jsonPath):
        #     makeDirsForFile(jsonPath)
        with open(jsonPath, 'w') as f:
            json.dump(data, f, sort_keys=True, indent=4, separators=(',', ': '))
        computing_time = (timer() - start_time)/60
        print('Computing Time: %s min' % computing_time)

        # with open(jsonPath, 'a') as file:
        #     x = json.dumps(matrix, indent=4)
        #     file.write(x)



    # for text_file in map(lambda textFilename: os.path.join(textDataDir, textFilename), fnmatch.filter(os.listdir(textDataDir), '*.txt')):
    #     jsonData = {}
    #     # Count the number of lines in the file to customize the number of customers
    #     size = sum(1 for line in open(text_file))
    #     with open(text_file) as f:
    #         for lineNum, line in enumerate(f, start=1):
    #             if lineNum in [2, 3, 4, 6, 7, 8, 9]:
    #                 pass
    #             elif lineNum == 1:
    #                 # <Instance name>
    #                 jsonData['instance_name'] = line.strip()
    #
    #             elif lineNum == 5:
    #                 # <Maximum vehicle number>, <Vehicle capacity>
    #                 values = line.strip().split()
    #                 jsonData['max_vehicle_number'] = int(values[0])
    #                 jsonData['vehicle_capacity'] = float(values[1])
    #             else:
    #                 # <Customer number>, <X coordinate>, <Y coordinate>, <Demand>, <Ready time>, <Due date>, <Service time>
    #                 values = line.strip().split()
    #                 jsonData['customer_%s' % values[0]] = {
    #                     'coordinates': {
    #                         'x': float(values[2]),
    #                         'y': float(values[3]),
    #                     },
    #                     'demand': float(values[4]),
    #                     # 'ready_time': float(values[4]),
    #                     # 'due_time': float(values[5]),
    #                     # 'service_time': float(values[6]),
    #                 }
    #     numOfCustomers = size - 9
    #     customers = ['customer_%d' % x for x in range(0, numOfCustomers)]
    #     jsonData['distance_matrix'] = [[__distance(jsonData[customer1], jsonData[customer2])
    #                                     for customer1 in customers] for customer2 in customers]
    #     jsonFilename = '%s.json' % jsonData['instance_name']
    #     jsonPathname = os.path.join(jsonDataDir, jsonFilename)
    #     print('Write to file: %s' % jsonPathname)
    #     makeDirsForFile(pathname=jsonPathname)
    #     with open(jsonPathname, 'w') as f:
    #         json.dump(jsonData, f, sort_keys=True, indent=4, separators=(',', ': '))


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
def visualization(foldername, threshold, algName, instanceName, instance, route, crossover, mutation, select, waitCost,
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
            print(i)
        i += 1
    df = pd.DataFrame(row)

    df.columns = ['customer_number', 'x_pos', 'y_pos']
    df.set_index('customer_number', inplace=True)
    plt.style.use('ggplot')
    # plotDataDir = os.path.join(BASE_DIR, 'plot', 'TEST', foldername, algName)
    plotDataDir = os.path.join(BASE_DIR, 'plot', 'CRZ', foldername, algName)
    # plotDataDir = os.path.join(BASE_DIR, 'plot', 'CRZ_split', foldername, algName)

    # try:
    #     os.makedirs(plotDataDir)
    # except:
    #     pass

    # Plot the scatter plot of customers
    # plt.figure()
    ax = df.plot.scatter(x='x_pos', y='y_pos', color='red', s=1, figsize=(10, 10))

    # Label the axis
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # define different colors for different routes
    # colors = cycle(["b", "r",  "g", "purple", "orange", "grey", "black", "pink", "y", "brown"])
    # colors = iter(plt.cm.rainbow(np.linspace(0,1,1000)))
    colors = iter(pl.cm.jet(np.linspace(0,1,1000)))
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

    # Save to folder
    currentTime = time.strftime("_%m-%d_%H-%M-%S")
    try:
        # parameters = '_cx%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_f%s_%s' % (crossover, mutation, select, waitCost,
        #                                                                          detourCost, indSize, popSize, cxPb, mutPb, threshold, no)
        parameters = '_cro%s_mut%s_sel%s_wC%s_dC%s_iS%s_pS%s_cP%s_mP%s_nG%s' % (crossover, mutation, select, waitCost,
                                                                                      detourCost, indSize, popSize, cxPb, mutPb, NGen)
    except:
        parameters = currentTime
    file_location = os.path.join(plotDataDir, instanceName)
    makeDirsForFile(pathname=file_location)
    plt.savefig(file_location + parameters + '.png')
