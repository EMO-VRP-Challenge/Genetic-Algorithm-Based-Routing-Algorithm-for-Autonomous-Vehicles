import glob, os
from src import BASE_DIR


# folder = os.path.join(BASE_DIR, 'benchmark', 'test', 'Uchoa')
folder = os.path.join(BASE_DIR, 'benchmark', 'test', 'Li')
# folder = os.path.join(BASE_DIR, 'benchmark', 'test', 'Cordeau-mdvrp')


def change_file_ext():
    for filename in glob.iglob(os.path.join(folder, '*.vrp')):
        os.rename(filename, filename[:-4] + '.txt')


def add_ext():
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.splitext(file)[1]:
            os.rename(path, path + '.txt')


def change_file_name():
    for pathAndFilename in glob.iglob(os.path.join(folder, '*.txt')):
        # title = os.path.splitext(os.path.basename(pathAndFilename)[0])
        # os.rename(pathAndFilename, os.path.join(title[2:] + '.txt'))
        # os.rename(pathAndFilename, pathAndFilename.replace('X\X', 'X'))
        os.rename(pathAndFilename, pathAndFilename.replace('Li\Li', 'Li'))


# change_file_ext()
add_ext()
# change_file_name()
#
#
#
# import math
# import random
#
# import matplotlib.pyplot as plt
#
# def select_point():
#     p = random.random()
#     if p < 0.5:
#         return 0
#     return 1
#
# def sample_point(R):
#     """
#     Sample point inpolar coordinates
#     """
#     phi = 2.0 * math.pi * random.random() # angle
#     r   = R * random.gauss(0.0, 1.0)      # might try different radial distribution, R*random.expovariate(1.0)
#
#     return (r * math.cos(phi), r * math.sin(phi))
#
# def sample(R, points):
#     idx = select_point()
#
#     x, y = sample_point(R)
#
#     return (x + points[idx][0], y + points[idx][1])
#
# R = 1.0
# points = [(7.1, 3.3), (4.8, -1.4)]
#
# random.seed(12345)
#
# xx = []
# yy = []
# cc = []
#
# xx.append(points[0][0])
# xx.append(points[1][0])
#
# yy.append(points[0][1])
# yy.append(points[1][1])
#
# cc.append(0.8)
# cc.append(0.8)
#
# for k in range(0, 50):
#     x, y = sample(R, points)
#     xx.append(x)
#     yy.append(y)
#     cc.append(0.3)
#
# plt.scatter(xx, yy, c=cc)
# plt.show()
