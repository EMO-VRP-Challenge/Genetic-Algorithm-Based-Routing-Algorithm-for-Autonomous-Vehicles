import random
import numpy as np
from GA import GA
import os
import glob
from src import BASE_DIR
from json import load

customer_no = 100
popSize = 1000 # 30
NGen = 250


def alg(algorithm):
    folder = os.path.join(BASE_DIR, 'benchmark')
    sfolder = os.path.join(folder, 'A')
    for filedir in glob.iglob(os.path.join(sfolder, '*.json')):
        instName = os.path.splitext(os.path.basename(filedir))[0]
        # for mutPb in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        #     for cxPb in [0.6, 0.7, 0.8]:
        for mutPb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
            for cxPb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #,1600, 1700,1800, 1900,2000
                # for popSize in [2000]: # 100, 500, 1000, 1500
                for popSize, NGen in zip([50000], [10]):
                # cxPb = 0.7
                # mutPb = 0.3
                #     start_time = timer()
                    with open(filedir) as in_f:
                        instance = load(in_f)

    running_veh = [[0, 1], [2, 3]]
    available_veh_count = 0
    vehicleCapacity = instance['vehicle_capacity']
    capacity_0 = vehicleCapacity - instance['customer_0']['demand']
    capacity_1 = vehicleCapacity - instance['customer_0']['demand']
    current_capacities = [capacity_0, capacity_1]

    for request_no in range(4, customer_no - 1, 2):
        demand = instance['customer_%d' % request_no]['demand']
        available_veh_id = []
        # available_veh = []
        for i, cap in enumerate(current_capacities[:]):
            if cap >= demand:
                available_veh_count += 1
                available_veh_id.append(i)
                # available_veh.append(running_veh[i])

        if available_veh_count == 0:
            # no running veh has sufficient cap, deploy a new veh
            running_veh.append([request_no, request_no + 1])
            current_capacities.append([vehicleCapacity - demand])
        elif algorithm == 'min_req':
            running_veh, current_capacities = min_req(running_veh, request_no, current_capacities, demand)
        elif algorithm == 'min_distance':
            running_veh, current_capacities = min_distance(running_veh, available_veh_id, request_no, current_capacities, demand, instance)
        elif algorithm == 'avg_distance':
            running_veh, current_capacities = avg_distance(running_veh, available_veh_id, request_no, current_capacities, demand, instance)
        elif algorithm == 'min_waiting':
            running_veh, current_capacities = min_waiting(running_veh, available_veh_id, request_no, current_capacities, demand, instance)
        elif algorithm == 'avg_waiting':
            running_veh, current_capacities = avg_waiting(running_veh, available_veh_id, request_no, current_capacities, demand, instance)
        elif algorithm == 'GA':
            running_veh, current_capacities = classical_GA(running_veh, available_veh_id, request_no, current_capacities, demand)
    return running_veh



def min_req(running_veh, request_no, current_capacities, demand):
    min_req = min(running_veh)
    min_len = len(min_req)
    min_req_id = [i for i, j in enumerate(running_veh) if len(j) == min_len]

    if len(min_req_id) == 1: # only one veh has enough cap to hold the new req
        running_veh[min_req_id[0]].append(request_no)
        running_veh[min_req_id[0]].append(request_no + 1)
        current_capacities[min_req_id[0]] -= demand
    else:
    # len(min_req_id) > 1:
        indSize = len(min_req_id)
        random.seed()
         # 30
         # '(fit0.0003)' #800
        exportCSV = True
        # exportCSV = False
        # customizeData = True
        customizeData = '2'
        chosen_veh_id = GA(
        instance,
        running_veh,
        min_req_id,
        request_no,
        # file=filedir,
        # instName=instName,
        # crossover='PM',
        crossover='Ord',
        # mutation='Inv',
        mutation='Shu',
        # select='Rou',
        select='Tour',
        indSize=indSize,
        popSize=popSize,
        # cxPb=cxPb,
        # mutPb=mutPb,
        NGen=NGen,
        # start_time=start_time,
        exportCSV=exportCSV,
        customizeData=customizeData
        )
        chosen_veh_index = min_req_id[chosen_veh_id]
        chosen_veh = running_veh[chosen_veh_index]
        chosen_veh.append(request_no)
        chosen_veh.append(request_no + 1)
        current_capacities[chosen_veh_index] -= demand
    return running_veh, current_capacities


def min_distance(running_veh, available_veh_id, request_no, current_capacities, demand, instance):
    distances_list = []

    available_veh = [running_veh[i] for i in available_veh_id]
    for index, list in enumerate(available_veh):
        for ind, veh in enumerate(list[:-1]):
            distance = instance['distance_matrix'][veh][list[ind + 1]]
            distances_list.append(distance)
    min_dist = min(distances_list)
    min_dist_id = [i for i, j in enumerate(distances_list) if j == min_dist]
    if len(min_dist_id) == 1:
        running_veh[available_veh_id[min_dist_id[0]]].append(request_no)
        running_veh[available_veh_id[min_dist_id[0]]].append(request_no + 1)
        current_capacities[available_veh_id[min_dist_id[0]]] -= demand
    else:
    # len(min_req_id) > 1:
        indSize = len(min_dist_id)
        random.seed()
         # 30
         # '(fit0.0003)' #800
        exportCSV = True
        # exportCSV = False
        # customizeData = True
        customizeData = '2'
        chosen_veh_id = GA(
        instance,
        available_veh,
        min_dist_id,
        request_no,
        # file=filedir,
        # instName=instName,
        # crossover='PM',
        crossover='Ord',
        # mutation='Inv',
        mutation='Shu',
        # select='Rou',
        select='Tour',
        indSize=indSize,
        popSize=popSize,
        # cxPb=cxPb,
        # mutPb=mutPb,
        NGen=NGen,
        # start_time=start_time,
        exportCSV=exportCSV,
        customizeData=customizeData
        )
        chosen_veh_index = available_veh_id[min_dist_id[chosen_veh_id]]
        chosen_veh = running_veh[chosen_veh_index]
        chosen_veh.append(request_no)
        chosen_veh.append(request_no + 1)
        current_capacities[chosen_veh_index] -= demand
    return running_veh, current_capacities


def avg_distance(running_veh, available_veh_id, request_no, current_capacities, demand, instance):
    distances_list = []
    available_veh = [running_veh[i] for i in available_veh_id]
    for index, list in enumerate(available_veh):
        for ind, veh in enumerate(list[:-1]):
            distance = instance['distance_matrix'][veh][list[ind + 1]]
            distances_list.append(distance)
    avg_dist = np.mean(distances_list)
    avg_dist_id = [i for i, j in enumerate(distances_list) if j <= avg_dist]
    if len(avg_dist_id) == 1:
        running_veh[available_veh_id[avg_dist_id[0]]].append(request_no)
        running_veh[available_veh_id[avg_dist_id[0]]].append(request_no + 1)
        current_capacities[available_veh_id[avg_dist_id[0]]] -= demand
    else:
    # len(min_req_id) > 1:
        indSize = len(avg_dist_id)
        random.seed()
         # 30
         # '(fit0.0003)' #800
        exportCSV = True
        # exportCSV = False
        # customizeData = True
        customizeData = '2'
        chosen_veh_id = GA(
        instance,
        available_veh,
        avg_dist_id,
        request_no,
        # file=filedir,
        # instName=instName,
        # crossover='PM',
        crossover='Ord',
        # mutation='Inv',
        mutation='Shu',
        # select='Rou',
        select='Tour',
        indSize=indSize,
        popSize=popSize,
        # cxPb=cxPb,
        # mutPb=mutPb,
        NGen=NGen,
        # start_time=start_time,
        exportCSV=exportCSV,
        customizeData=customizeData
        )
        chosen_veh_index = available_veh_id[avg_dist_id[chosen_veh_id]]
        chosen_veh = running_veh[chosen_veh_index]
        chosen_veh.append(request_no)
        chosen_veh.append(request_no + 1)
        current_capacities[chosen_veh_index] -= demand
    return running_veh, current_capacities


def min_waiting(running_veh, available_veh_id, request_no, current_capacities, demand, instance):
    waiting_time = []
    available_veh = [running_veh[i] for i in available_veh_id]
    for list in available_veh:
        waiting = instance['distance_matrix'][list[-1]][request_no]
        waiting_time.append(waiting)
    min_wait = min(waiting_time)
    min_wait_id = [i for i, j in enumerate(waiting_time) if j == min_wait]
    if len(min_wait_id) == 1:
        running_veh[available_veh_id[min_wait_id[0]]].append(request_no)
        running_veh[available_veh_id[min_wait_id[0]]].append(request_no + 1)
        current_capacities[available_veh_id[min_wait_id[0]]] -= demand
    else:
    # len(min_req_id) > 1:
        indSize = len(min_wait_id)
        random.seed()
         # 30
         # '(fit0.0003)' #800
        exportCSV = True
        # exportCSV = False
        # customizeData = True
        customizeData = '2'
        chosen_veh_id = GA(
        instance,
        available_veh,
        min_wait_id,
        request_no,
        # file=filedir,
        # instName=instName,
        # crossover='PM',
        crossover='Ord',
        # mutation='Inv',
        mutation='Shu',
        # select='Rou',
        select='Tour',
        indSize=indSize,
        popSize=popSize,
        # cxPb=cxPb,
        # mutPb=mutPb,
        NGen=NGen,
        # start_time=start_time,
        exportCSV=exportCSV,
        customizeData=customizeData
        )
        chosen_veh_index = available_veh_id[min_wait_id[chosen_veh_id]]
        chosen_veh = running_veh[chosen_veh_index]
        chosen_veh.append(request_no)
        chosen_veh.append(request_no + 1)
        current_capacities[chosen_veh_index] -= demand
    return running_veh, current_capacities


def avg_waiting(running_veh, available_veh_id, request_no, current_capacities, demand, instance):
    waiting_time = []
    available_veh = [running_veh[i] for i in available_veh_id]
    for list in available_veh:
        waiting = instance['distance_matrix'][list[-1]][request_no]
        waiting_time.append(waiting)
    avg_wait = np.mean(waiting_time)
    avg_wait_id = [i for i, j in enumerate(waiting_time) if j <= avg_wait]
    if len(avg_wait_id) == 1:
        running_veh[available_veh_id[avg_wait_id[0]]].append(request_no)
        running_veh[available_veh_id[avg_wait_id[0]]].append(request_no + 1)
        current_capacities[available_veh_id[avg_wait_id[0]]] -= demand
    else:
    # len(min_req_id) > 1:
        indSize = len(avg_wait_id)
        random.seed()
         # 30
         # '(fit0.0003)' #800
        exportCSV = True
        # exportCSV = False
        # customizeData = True
        customizeData = '2'
        chosen_veh_id = GA(
        instance,
        available_veh,
        avg_wait_id,
        request_no,
        # file=filedir,
        # instName=instName,
        # crossover='PM',
        crossover='Ord',
        # mutation='Inv',
        mutation='Shu',
        # select='Rou',
        select='Tour',
        indSize=indSize,
        popSize=popSize,
        # cxPb=cxPb,
        # mutPb=mutPb,
        NGen=NGen,
        # start_time=start_time,
        exportCSV=exportCSV,
        customizeData=customizeData
        )
        chosen_veh_index = available_veh_id[avg_wait_id[chosen_veh_id]]
        chosen_veh = running_veh[chosen_veh_index]
        chosen_veh.append(request_no)
        chosen_veh.append(request_no + 1)
        current_capacities[chosen_veh_index] -= demand
    return running_veh, current_capacities


def classical_GA(running_veh, available_veh_id, request_no, current_capacities, demand):
    available_veh = [running_veh[i] for i in available_veh_id]
    indSize = len(available_veh)
    random.seed()
     # 30
     # '(fit0.0003)' #800
    exportCSV = True
    # exportCSV = False
    # customizeData = True
    customizeData = '2'
    chosen_veh_id = GA(
    instance,
    available_veh,
    0,
    request_no,
    # file=filedir,
    # instName=instName,
    # crossover='PM',
    crossover='Ord',
    # mutation='Inv',
    mutation='Shu',
    # select='Rou',
    select='Tour',
    indSize=indSize,
    popSize=popSize,
    # cxPb=cxPb,
    # mutPb=mutPb,
    NGen=NGen,
    # start_time=start_time,
    exportCSV=exportCSV,
    customizeData=customizeData
    )
    chosen_veh = running_veh[chosen_veh_id]
    chosen_veh.append(request_no)
    chosen_veh.append(request_no + 1)
    current_capacities[chosen_veh_id] -= demand
    return running_veh, current_capacities


def main():
    algorithm = 'GA'

    alg(algorithm)
