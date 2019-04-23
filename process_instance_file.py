# -*- coding: utf-8 -*-
# process_instance_file.py
import glob
import os
from src import BASE_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.datasets.samples_generator import make_blobs
import math
from src.utils import makeDirsForFile

# folder = os.path.join(BASE_DIR, 'results', 'CRZ', 'p100g200', 'Z')
# folder = os.path.join(BASE_DIR, 'results', 'CRZ_split', '2', 'C')
# sfolder = os.path.join(BASE_DIR, 'benchmark', 'CRZ_split', 'Zones2')
# folder = os.path.join(BASE_DIR, 'benchmark', 'CRZ', 'Zones')
folder = os.path.join(BASE_DIR, 'benchmark', 'CRZ')
# folder = os.path.join(BASE_DIR, 'benchmark', 'CRZ', 'Random')
# sfolder = os.path.join(BASE_DIR, 'benchmark', 'CRZ-1_demand', 'Zones2')
# print(folder)

sfolder = os.path.join(folder, 'fig')
os.makedirs(sfolder, exist_ok=True)

# file = 'Solomon.py'
# new_dir = os.path.join(BASE_DIR, 'src')
# new_dir2 = os.path.join(BASE_DIR, 'plot')
# os.rename(os.path.join(new_dir, file), os.path.join(new_dir2, file))

def check_min_max():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        # for i in range(1200):
        for i in range(1001):
            if float(rows[i][1]) < xmin:
                xmin = float(rows[i][1])
            if float(rows[i][2]) < ymin:
                ymin = float(rows[i][2])
            if float(rows[i][1]) > xmax:
                xmax = float(rows[i][1])
            if float(rows[i][2]) > ymax:
                ymax = float(rows[i][2])
        print(xmin, xmax)
        print(ymin, ymax)


def plot_instance():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        xy = []
        # for i in range(1200):
        for i in range(2000):
            # xy.append([float(rows[i][1]), float(rows[i][2])])
            xy.append([int(rows[i][1]), int(rows[i][2])])
        df = pd.DataFrame(xy)
        df.columns = ['x', 'y']
        fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
        # fig.savefig(sfolder + '/Li_32.png')
        fig.savefig(sfolder + '/X-n1001-k43_2000.png')


def check_zones():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        zone1 = 0
        zone2 = 0
        zone3 = 0
        # for i in range(1200):
        for i in range(1001):
            if np.sqrt(float(rows[i][1]) ** 2 + float(rows[i][2]) ** 2) <= 100:
                zone1 += 1
            elif 100 < np.sqrt(float(rows[i][1]) ** 2 + float(rows[i][2]) ** 2) <= 250:
                zone2 += 1
            elif 250 < np.sqrt(float(rows[i][1]) ** 2 + float(rows[i][2]) ** 2) <= 500:
                zone3 += 1
        print(zone1, zone2, zone3)


# def format_instance():
#     for filedir in glob.iglob(os.path.join(folder, '*.txt')):
#         with open(filedir, 'r') as f:
#             lines = f.readlines()
#         rows = []
#         for line in lines:
#             rows.append(line.split(' '))
#         # rows = rows[10 : 2411]
#         rows = rows[7 : 2010]
#         # rows = rows[:1001]
#
#         # for lineNum, line in enumerate(f, start=1):
#         #     for lineNum in range(9, 109):
#         #         values = line.strip().split()
#         # j = 0
#         # for i in range(9, 109):
#         #     rows[i][0] = str(j)
#         #     j += 1
#         with open(filedir, "w") as output_file:
#             for line in rows:
#                 output_file.write(' '.join([str(x) for x in line]))


def tab_to_space():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        instName = os.path.splitext(os.path.basename(filedir))[0]
        with open(filedir) as fin, open(os.path.join(sfolder, instName+'.txt'), 'w') as fout:
            for line in fin:
                fout.write(line.replace('\t', ' '))


# first run, afterwards replace all \t with spaces
def format_instance_0():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        instName = os.path.splitext(os.path.basename(filedir))[0]
        req_nr = int(instName[3:6]) # 1001
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        rows = rows[7 : 8 + 2 * req_nr] # 2010 = 7 + 2 * req_nr + 1
        with open(filedir, "w") as output_file:
            for line in rows:
                output_file.write(' '.join([str(x) for x in line]))


def format_instance():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        instName = os.path.splitext(os.path.basename(filedir))[0]
        req_nr = int(instName[3:6]) # 1001
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))

        # rows = rows[7 : 8 + 2 * req_nr] # 2010 = 7 + 2 * req_nr + 1

        for i in range(req_nr):
            rows[i][2] = rows[i][2][:-1] # remove the '\n' at the end of rows[][2]
            demand = rows[req_nr+1+i][1]
            if '\n' not in demand:
                demand = demand + '\n'
            rows[i].append(demand)
        rows = rows[:req_nr]

        rows[0][3] = str(random.randint(1, 100)) + '\n' # coz by default the demand of psg 1 is 0
        with open(filedir, "w") as output_file:
            for line in rows:
                output_file.write(' '.join([str(x) for x in line]))


def add_coord():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        instName = os.path.splitext(os.path.basename(filedir))[0]
        req_nr = int(instName[3:6]) # 1001
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        for i in range(2000 - req_nr):
            # x = random.uniform(0, 1000)
            # y = random.uniform(0, 1000)
            x = random.randint(1, 1000)
            y = random.randint(1, 1000)
            d = random.randint(1, 100)
            rows.append([str(req_nr + 1 + i), str(x), str(y), str(d)+'\n'])
        with open(filedir, "w") as output_file:
            for line in rows:
                output_file.write(' '.join([str(x) for x in line]))


def gen_blobs():
    X, y = make_blobs(n_samples=100, n_features=2, centers=3)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.savefig(sfolder + '/cluster.png')


def gen_clusters():
    points = []
    centers = []
    center_nr = random.randint(2, 4)

    # center_x0 = random.randint(100, 900)
    # center_y0 = random.randint(100, 900)
    # center = [center_x0, center_y0]
    # centers.append(center)
    # for i in range(1, center_nr):
    #     center_alpha = 2 * math.pi * random.random() # angle
    #     center_distance = random.uniform(400, 600)
    #     r = center_distance * np.sqrt(random.random())
    #     center_x2 = r * math.cos(center_alpha) + centers[i-1][0]
    #     center_y2 = r * math.sin(center_alpha) + centers[i-1][1]
    #     center2 = [center_x2, center_y2]
    #     centers.append(center2)
    for i in range(center_nr):
        center_x0 = random.randint(100, 900)
        center_y0 = random.randint(100, 900)
        center = [center_x0, center_y0]
        centers.append(center)

    if center_nr == 2:
        while np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2) < 400:
            centers = []
            for _ in range(center_nr):
                center_x = random.randint(100, 900)
                center_y = random.randint(100, 900)
                center = [center_x, center_y]
                centers.append(center)

        # centers = []
        # center_x1 = random.randint(100, 900)
        # center_y1 = random.randint(100, 900)
        # center1 = [center_x1, center_y1]
        # centers.append(center1)

        first_cluster = random.randrange(400, 1601, 2)
        point_limit = first_cluster
        while point_limit > 0:
            deviationFromPoint = random.uniform(100, 400)
            # newCoords = [centers[0][i] + random.random() * deviationFromPoint for i in range(2)]

            # x = random.randint(0, 1000)
            # y = random.randint(0, 1000)
            # while np.sqrt((x-centers[0][0]) ** 2 + (y-centers[0][1]) ** 2) > deviationFromPoint:
            #     x = random.randint(0, 1000)
            #     y = random.randint(0, 1000)

            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[0][0])
            y = int(r * math.sin(alpha) + centers[0][1])
            if 0 < x < 1000 and 0 < y < 1000:
                point_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
        # for _ in range(1000 - point_limit):
        sec_cluster = 2000 - first_cluster
        mid_limit = sec_cluster
        while mid_limit > 0:
            deviationFromPoint = random.uniform(100, 300)
            # newCoords = [centers[1][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[1][0])
            y = int(r * math.sin(alpha) + centers[1][1])
            if 0 < x < 1000 and 0 < y < 1000:
                mid_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)


    elif center_nr == 3:
        while np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2) < 400 \
                or np.sqrt((centers[0][0] - centers[2][0]) ** 2 + (centers[0][1] - centers[2][1]) ** 2) < 400 \
                or np.sqrt((centers[1][0] - centers[2][0]) ** 2 + (centers[1][1] - centers[2][1]) ** 2) < 400:
            centers = []
            for _ in range(center_nr):
                center_x = random.randint(100, 900)
                center_y = random.randint(100, 900)
                center = [center_x, center_y]
                centers.append(center)

        first_cluster = random.randrange(400, 1001, 2)
        point_limit = first_cluster
        while point_limit > 0:
            deviationFromPoint = random.uniform(100, 300)
            # newCoords = [centers[0][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[0][0])
            y = int(r * math.sin(alpha) + centers[0][1])
            if 0 < x < 1000 and 0 < y < 1000:
                point_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
        sec_cluster = random.randrange(200, 801, 2)
        mid_limit = sec_cluster
        while mid_limit > 0:
            deviationFromPoint = random.uniform(100, 200)
            # newCoords = [centers[1][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[1][0])
            y = int(r * math.sin(alpha) + centers[1][1])
            if 0 < x < 1000 and 0 < y < 1000:
                mid_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
        mid_limit2 = 2000 - first_cluster - sec_cluster
        while mid_limit2 > 0:
            deviationFromPoint = random.uniform(100, 300)
            # newCoords = [centers[2][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[2][0])
            y = int(r * math.sin(alpha) + centers[2][1])
            if 0 < x < 1000 and 0 < y < 1000:
                mid_limit2 -= 1
                newCoords = [x, y]
                points.append(newCoords)

    else:
        while np.sqrt((centers[0][0] - centers[1][0]) ** 2 + (centers[0][1] - centers[1][1]) ** 2) < 300 \
                or np.sqrt((centers[0][0] - centers[2][0]) ** 2 + (centers[0][1] - centers[2][1]) ** 2) < 300 \
                or np.sqrt((centers[1][0] - centers[2][0]) ** 2 + (centers[1][1] - centers[2][1]) ** 2) < 300 \
                or np.sqrt((centers[0][0] - centers[3][0]) ** 2 + (centers[0][1] - centers[3][1]) ** 2) < 300 \
                or np.sqrt((centers[1][0] - centers[3][0]) ** 2 + (centers[1][1] - centers[3][1]) ** 2) < 300 \
                or np.sqrt((centers[2][0] - centers[3][0]) ** 2 + (centers[2][1] - centers[3][1]) ** 2) < 300:
            centers = []
            for _ in range(center_nr):
                center_x = random.randint(100, 900)
                center_y = random.randint(100, 900)
                center = [center_x, center_y]
                centers.append(center)

        first_cluster = random.randrange(200, 601, 2)
        point_limit = first_cluster
        while point_limit > 0:
            deviationFromPoint = random.uniform(100, 200)
            # newCoords = [centers[0][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[0][0])
            y = int(r * math.sin(alpha) + centers[0][1])
            if 0 < x < 1000 and 0 < y < 1000:
                point_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
        sec_cluster = random.randrange(400, 601, 2)
        mid_limit = sec_cluster
        while mid_limit > 0:
            deviationFromPoint = random.uniform(100, 250)
            # newCoords = [centers[1][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[1][0])
            y = int(r * math.sin(alpha) + centers[1][1])
            if 0 < x < 1000 and 0 < y < 1000:
                mid_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
        third_cluster = random.randrange(200, 601, 2)
        mid_limit2 = third_cluster
        while mid_limit2 > 0:
            deviationFromPoint = random.uniform(100, 300)
            # newCoords = [centers[2][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[2][0])
            y = int(r * math.sin(alpha) + centers[2][1])
            if 0 < x < 1000 and 0 < y < 1000:
                mid_limit2 -= 1
                newCoords = [x, y]
                points.append(newCoords)
        mid_limit3 = 2000 - first_cluster - sec_cluster - third_cluster
        while mid_limit3 > 0:
            deviationFromPoint = random.uniform(100, 300)
            # newCoords = [centers[3][i] + random.random() * deviationFromPoint for i in range(2)]
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[3][0])
            y = int(r * math.sin(alpha) + centers[3][1])
            if 0 < x < 1000 and 0 < y < 1000:
                mid_limit3 -= 1
                newCoords = [x, y]
                points.append(newCoords)
    print(centers)
    df = pd.DataFrame(points)
    df.columns = ['x', 'y']
    demand = []
    for _ in range(2000):
        demand.append(random.randint(1, 4))
    df['demand'] = demand
    name = 'C13'
    path_out = os.path.join(folder, name + '.csv')
    df.to_csv(path_out, index=True)
    fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
    # fig.savefig(sfolder + '/Li_32.png')
    fig.savefig(sfolder + '/' + name + '.png')


def gen_4_clusters():
    # for ite in range(no):
    centers = []
    x = random.uniform(200, 300)
    y = random.uniform(700, 800)
    centers.append([x, y])
    x = random.uniform(200, 300)
    y = random.uniform(200, 300)
    centers.append([x, y])
    x = random.uniform(700, 800)
    y = random.uniform(700, 800)
    centers.append([x, y])
    x = random.uniform(700, 800)
    y = random.uniform(200, 300)
    centers.append([x, y])
    points = []
    for i in range(4):
        point_limit = 50 # 50
        while point_limit > 0:
            deviationFromPoint = random.uniform(100, 200)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[i][0])
            y = int(r * math.sin(alpha) + centers[i][1])
            if 0 < x < 1000 and 0 < y < 1000:
                point_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
    df = pd.DataFrame(points)
    df.columns = ['x', 'y']
    demand = []
    for _ in range(200): # 200
        demand.append(random.randint(1, 4))
    df['demand'] = demand
    # name = f'100req-C{ite}'
    name = '100req-C2'
    path_out = os.path.join(folder, name + '.csv')
    # df.columns = ['ID','x', 'y', 'demand']
    df.to_csv(path_out, index=True)
    fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
    # fig.savefig(sfolder + '/Li_32.png')
    fig.savefig(folder + '/' + name + '.png')


def test():
    # radius of the circle
    circle_r = 300
    # center of the circle (x, y)
    circle_x = 500
    circle_y = 700
    points = []
    for _ in range(1000):
        # random angle
        alpha = 2 * math.pi * random.random()
        # random radius
        r = circle_r * np.sqrt(random.random())
        # calculating coordinates
        x = r * math.cos(alpha) + circle_x
        y = r * math.sin(alpha) + circle_y
        newCoords = [x, y]
        points.append(newCoords)
    # print(centers)
    df = pd.DataFrame(points)
    df.columns = ['x', 'y']
    fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
    # fig.savefig(sfolder + '/Li_32.png')
    fig.savefig(sfolder + '/test.png')


def gen_zones(iter):
    for i in range(iter):
        center = [500, 500]
        zone1 = []
        zone2 = []
        zone3 = []
        # for _ in range(400):
        for _ in range(40):
            deviationFromPoint = random.uniform(0, 100)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + center[0])
            y = int(r * math.sin(alpha) + center[1])
            zone1.append([x, y])

        for _ in range(30):
            deviationFromPoint = random.uniform(100, 300)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + center[0])
            y = int(r * math.sin(alpha) + center[1])
            zone2.append([x, y])

            deviationFromPoint = random.uniform(0, 100)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + center[0])
            y = int(r * math.sin(alpha) + center[1])
            zone2.append([x, y])
        for _ in range(50):
            deviationFromPoint = random.uniform(300, 500)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + center[0])
            y = int(r * math.sin(alpha) + center[1])
            zone3.append([x, y])

            deviationFromPoint = random.uniform(0, 100)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + center[0])
            y = int(r * math.sin(alpha) + center[1])
            zone3.append([x, y])

        # print(len(zone1), len(zone2), len(zone3))
        zones = zone1 + zone2 + zone3
        df = pd.DataFrame(zones)
        df.columns = ['x', 'y']
        demand = []
        for _ in range(200):
            demand.append(random.randint(1, 4))
        df['demand'] = demand
        name = '100req-Z' + str(i+1)
        path_out = os.path.join(folder, name + '.csv')
        df.to_csv(path_out, index=True)
        fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
        # fig.savefig(sfolder + '/Li_32.png')
        fig.savefig(folder + '/' + name + '_plot.png')


def change_demands_in_R_set():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        for i in range(200):
            rows[i][3] = str(int(rows[i][3]) % 4 + 1) + '\n'
        with open(filedir, "w") as output_file:
            for line in rows:
                output_file.write(' '.join([str(x) for x in line]))


def split_demands():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        for i in range(0, 2000, 2):
            demand = int(rows[i][3][:-1])
            if demand > 1:
                rows[i][3] = '1\n'
                for _ in range(demand-1):
                    rows.append(rows[i])
                    rows.append(rows[i+1])
        with open(filedir, "w") as output_file:
            for line in rows:
                output_file.write(' '.join([str(x) for x in line]))


def csv_split_demands():
    for filedir in glob.iglob(os.path.join(folder, '*.csv')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        df = pd.read_csv(filedir)
        for i in range(0, 2000, 2):
            demand = df.at[i, 'demand']
            if demand > 1:
                df.at[i, 'demand'] = 1
        #         for _ in range(demand-1):
        #             df = df.append(df.iloc[i], ignore_index=True)
        #             df = df.append(df.iloc[i+1], ignore_index=True)
        # df.ID = [i for i in range(df.shape[0])]
        path_out = os.path.join(sfolder, name + '-1_demand.csv')
        df.to_csv(path_out, index=False)


def dropoff_demand_change_txt():
    for filedir in glob.iglob(os.path.join(folder, '*.txt')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        with open(filedir, 'r') as f:
            lines = f.readlines()
        rows = []
        for line in lines:
            rows.append(line.split(' '))
        for i in range(1, 2000, 2):
            # demand = int(rows[i][3][:-1])
            rows[i][3] = '0\n'

        with open(filedir, "w") as output_file:
            for line in rows:
                output_file.write(' '.join([str(x) for x in line]))

        for i in range(2000):
            for j in range(3):
                rows[i][j] = int(rows[i][j])
            rows[i][3] = int(rows[i][3][:-1])
        df = pd.DataFrame(rows)
        df.columns = ['no','x', 'y', 'demand']
        path_out = os.path.join(folder, name + '.csv')
        df.to_csv(path_out, index=False)


def dropoff_demand_change_csv():
    for filedir in glob.iglob(os.path.join(folder, '*.csv')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        df = pd.read_csv(filedir)
        for i in range(1, df.shape[0], 2):
            df.at[i, 'demand'] = 0
        df.drop(df.columns[0], axis=1, inplace=True)
        path_out = os.path.join(folder, name + '.csv')
        df.to_csv(path_out, index=True)


def line_connect_pickup_dropoff():
    for filedir in glob.iglob(os.path.join(folder, '*.csv')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        df = pd.read_csv(filedir)
        for i in range(100, 200, 2):
            plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'bo-', linewidth=1, markersize=2, alpha=0.5) #, label='zone C')
        for i in range(40, 100, 2):
            plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'go-', linewidth=1, markersize=2, alpha=0.5) #, label='zone B')
        for i in range(0, 40, 2):
            plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'ro-', linewidth=1, markersize=2, alpha=0.5) #, label='zone A')
        # for i in range(1500, 2000, 2):
        #     plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'yo-', linewidth=1, markersize=1, alpha=0.1)
        # for i in range(1000, 1500, 2):
        #     plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'bo-', linewidth=1, markersize=1, alpha=0.1)
        # for i in range(500, 1000, 2):
        #     plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'go-', linewidth=1, markersize=1, alpha=0.1)
        # for i in range(0, 500, 2):
        #     plt.plot([df.iloc[i, 1], df.iloc[i+1, 1]], [df.iloc[i, 2], df.iloc[i+1, 2]], 'ro-', linewidth=1, markersize=1, alpha=0.1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.savefig(sfolder + '/' + name + '.png')


def plot_from_csv():
    plt.style.use('ggplot')
    for filedir in glob.iglob(os.path.join(folder, '*.csv')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        df = pd.read_csv(filedir, index_col=0)
        # fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
        plot = df.plot.line(colormap='rainbow', figsize=(10,10), logy=True, grid=True)
        plot.set(xlabel='algorithm', ylabel='fitness')
        fig = plot.get_figure()
        fig.savefig(folder + '/' + name + '.png')


def rename_first_col_name():
    for filedir in glob.iglob(os.path.join(folder, '*.csv')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        df = pd.read_csv(filedir)
        # df.rename(columns={df.columns[0]: "ID"})
        df.columns = ['ID','x', 'y', 'demand']
        path_out = os.path.join(folder, name + '.csv')
        df.to_csv(path_out, index=False)


def zones2():
    folder2 = os.path.join(BASE_DIR, 'benchmark', 'CRZ', 'Zones2')
    # os.makedirs(folder2)
    for filedir in glob.iglob(os.path.join(folder, '*.csv')):
        name = os.path.splitext(os.path.basename(filedir))[0]
        df = pd.read_csv(filedir)
        for i in range(0, 2000, 2):
            pickup, dropoff = df.iloc[i].copy(), df.iloc[i+1].copy()
            df.iloc[i], df.iloc[i+1] = dropoff, pickup
        # df.set_index('ID', inplace=True)
        # df.reset_index(drop=True, inplace=True)
        # df.columns = ['ID','x', 'y', 'demand']
        df.ID = [j for j in range(2000)]
        path_out = os.path.join(folder2, name + '.csv')
        df.to_csv(path_out, index=False)


def gen_rand(no):
    for ite in range(no):
        points = []
        for i in range(200):
            x = random.randint(1, 999)
            y = random.randint(1, 999)
            newCoords = [x, y]
            points.append(newCoords)
        df = pd.DataFrame(points)
        df.columns = ['x', 'y']
        demand = []
        for _ in range(200): # 200
            demand.append(random.randint(1, 4))

        df['demand'] = demand
        # name = f'100req-C{ite}'
        name = f'100req-R{ite}'
        path_out = os.path.join(folder, name + '.csv')
        # df.columns = ['ID','x', 'y', 'demand']
        df.to_csv(path_out, index=True)
        fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
        # fig.savefig(sfolder + '/Li_32.png')
        fig.savefig(sfolder + '/' + name + '.png')


def gen_demand_specified_rand(no):
    for ite in range(no):
        points = []
        for i in range(200):
            x = random.randint(1, 999)
            y = random.randint(1, 999)
            newCoords = [x, y]
            points.append(newCoords)
        df = pd.DataFrame(points)
        df.columns = ['x', 'y']
        demand = []
        for _ in range(25): # 200
            demand.append(random.randint(2, 4))
        demand.extend([1]*75)
        random.shuffle(demand)
        ind = 1
        while ind <= len(demand):
            demand.insert(ind, 0)
            ind += 2
        df['demand'] = demand
        # name = f'100req-C{ite}'
        name = f'100req-dmd1-R{ite}'
        path_out = os.path.join(folder, name + '.csv')
        # df.columns = ['ID','x', 'y', 'demand']
        df.to_csv(path_out, index=True)
        fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
        # fig.savefig(sfolder + '/Li_32.png')
        fig.savefig(sfolder + '/' + name + '.png')



def gen_demand_specified_clusters():
    # for ite in range(no):
    centers = []
    x = random.uniform(200, 300)
    y = random.uniform(700, 800)
    centers.append([x, y])
    x = random.uniform(200, 300)
    y = random.uniform(200, 300)
    centers.append([x, y])
    x = random.uniform(700, 800)
    y = random.uniform(700, 800)
    centers.append([x, y])
    x = random.uniform(700, 800)
    y = random.uniform(200, 300)
    centers.append([x, y])
    points = []
    for i in range(4):
        point_limit = 50 # 50
        while point_limit > 0:
            deviationFromPoint = random.uniform(100, 200)
            alpha = 2 * math.pi * random.random()
            r = deviationFromPoint * np.sqrt(random.random())
            x = int(r * math.cos(alpha) + centers[i][0])
            y = int(r * math.sin(alpha) + centers[i][1])
            if 0 < x < 1000 and 0 < y < 1000:
                point_limit -= 1
                newCoords = [x, y]
                points.append(newCoords)
    df = pd.DataFrame(points)
    df.columns = ['x', 'y']
    demand = []
    for _ in range(25): # 200
        demand.append(random.randint(2, 4))
    demand.extend([1]*75)
    random.shuffle(demand)
    ind = 1
    while ind <= len(demand):
        demand.insert(ind, 0)
        ind += 2
    df['demand'] = demand
    # name = f'100req-C{ite}'
    name = '100req-dmd1-C1'
    path_out = os.path.join(folder, name + '.csv')
    # df.columns = ['ID','x', 'y', 'demand']
    df.to_csv(path_out, index=True)
    fig = df.plot.scatter(x='x', y='y', s=1).get_figure()
    # fig.savefig(sfolder + '/Li_32.png')
    fig.savefig(folder + '/' + name + '.png')


# format_instance_0()
# tab_to_space()
# format_instance()
# add_coord()
# plot_instance()
# check_min_max()
# check_zones()
# gen_clusters()
# test()
# gen_zones(30)
# change_demands_in_R_set()
# split_demands()
# dropoff_demand_change_csv()
# gen_4_clusters()
# line_connect_pickup_dropoff()
# plot_from_csv()
# gen_rand(4)
# gen_demand_specified_clusters()
gen_demand_specified_rand(4)
rename_first_col_name()
# zones2()
# csv_split_demands()
