import glob
import os
from src import BASE_DIR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns



def sharing0():

    C = []
    R = []
    Z = []
    Zi = []

    C1 = []
    C2 = []
    C3 = []
    C4 = []
    C5 = []
    C6 = []
    C7 = []
    C8 = []

    R1 = []
    R2 = []
    R3 = []
    R4 = []
    R5 = []
    R6 = []
    R7 = []
    R8 = []

    Z1 = []
    Z2 = []
    Z3 = []
    Z4 = []
    Z5 = []
    Z6 = []
    Z7 = []
    Z8 = []

    Zi1 = []
    Zi2 = []
    Zi3 = []
    Zi4 = []
    Zi5 = []
    Zi6 = []
    Zi7 = []
    Zi8 = []
    A = []
    for resultFile in glob.iglob(os.path.join(folder, '*.csv')):
        filename = os.path.splitext(os.path.basename(resultFile))[0]
        df = pd.read_csv(resultFile)
        if '-C' in filename:
            if 'alg1' in filename:
                C1.append(df.ix[0, 'avg_sharing'])
            elif 'alg2' in filename:
                C2.append(df.ix[0, 'avg_sharing'])
            elif 'alg3' in filename:
                C3.append(df.ix[0, 'avg_sharing'])
            elif 'alg4' in filename:
                C4.append(df.ix[0, 'avg_sharing'])
            elif 'alg5' in filename:
                C5.append(df.ix[0, 'avg_sharing'])
            elif 'alg6' in filename:
                C6.append(df.ix[0, 'avg_sharing'])
            elif 'alg7' in filename:
                C7.append(df.ix[0, 'avg_sharing'])
            elif 'alg8' in filename:
                C8.append(df.ix[0, 'avg_sharing'])
        elif '-R' in filename:
            if 'alg1' in filename:
                R1.append(df.ix[0, 'avg_sharing'])
            elif 'alg2' in filename:
                R2.append(df.ix[0, 'avg_sharing'])
            elif 'alg3' in filename:
                R3.append(df.ix[0, 'avg_sharing'])
            elif 'alg4' in filename:
                R4.append(df.ix[0, 'avg_sharing'])
            elif 'alg5' in filename:
                R5.append(df.ix[0, 'avg_sharing'])
            elif 'alg6' in filename:
                R6.append(df.ix[0, 'avg_sharing'])
            elif 'alg7' in filename:
                R7.append(df.ix[0, 'avg_sharing'])
            elif 'alg8' in filename:
                R8.append(df.ix[0, 'avg_sharing'])
        elif '_i_' in filename:
            if 'alg1' in filename:
                Zi1.append(df.ix[0, 'avg_sharing'])
            elif 'alg2' in filename:
                Zi2.append(df.ix[0, 'avg_sharing'])
            elif 'alg3' in filename:
                Zi3.append(df.ix[0, 'avg_sharing'])
            elif 'alg4' in filename:
                Zi4.append(df.ix[0, 'avg_sharing'])
            elif 'alg5' in filename:
                Zi5.append(df.ix[0, 'avg_sharing'])
            elif 'alg6' in filename:
                Zi6.append(df.ix[0, 'avg_sharing'])
            elif 'alg7' in filename:
                Zi7.append(df.ix[0, 'avg_sharing'])
            elif 'alg8' in filename:
                Zi8.append(df.ix[0, 'avg_sharing'])
        elif '-Z' in filename:
            if 'alg1' in filename:
                Z1.append(df.ix[0, 'avg_sharing'])
            elif 'alg2' in filename:
                Z2.append(df.ix[0, 'avg_sharing'])
            elif 'alg3' in filename:
                Z3.append(df.ix[0, 'avg_sharing'])
            elif 'alg4' in filename:
                Z4.append(df.ix[0, 'avg_sharing'])
            elif 'alg5' in filename:
                Z5.append(df.ix[0, 'avg_sharing'])
            elif 'alg6' in filename:
                Z6.append(df.ix[0, 'avg_sharing'])
            elif 'alg7' in filename:
                Z7.append(df.ix[0, 'avg_sharing'])
            elif 'alg8' in filename:
                Z8.append(df.ix[0, 'avg_sharing'])
    C.append(np.mean(C1))
    C.append(np.mean(C2))
    C.append(np.mean(C3))
    C.append(np.mean(C4))
    C.append(np.mean(C5))
    C.append(np.mean(C6))
    C.append(np.mean(C7))
    C.append(np.mean(C8))

    R.append(np.mean(R1))
    R.append(np.mean(R2))
    R.append(np.mean(R3))
    R.append(np.mean(R4))
    R.append(np.mean(R5))
    R.append(np.mean(R6))
    R.append(np.mean(R7))
    R.append(np.mean(R8))

    Z.append(np.mean(Z1))
    Z.append(np.mean(Z2))
    Z.append(np.mean(Z3))
    Z.append(np.mean(Z4))
    Z.append(np.mean(Z5))
    Z.append(np.mean(Z6))
    Z.append(np.mean(Z7))
    Z.append(np.mean(Z8))

    Zi.append(np.mean(Zi1))
    Zi.append(np.mean(Zi2))
    Zi.append(np.mean(Zi3))
    Zi.append(np.mean(Zi4))
    Zi.append(np.mean(Zi5))
    Zi.append(np.mean(Zi6))
    Zi.append(np.mean(Zi7))
    Zi.append(np.mean(Zi8))

    A.append(np.mean(C1+R1+Z1+Zi1))
    A.append(np.mean(C2+R2+Z2+Zi2))
    A.append(np.mean(C3+R3+Z3+Zi3))
    A.append(np.mean(C4+R4+Z4+Zi4))
    A.append(np.mean(C5+R5+Z5+Zi5))
    A.append(np.mean(C6+R6+Z6+Zi6))
    A.append(np.mean(C7+R7+Z7+Zi7))
    A.append(np.mean(C8+R8+Z8+Zi8))
    # aggregate_list.append(alg)
    alg = [C, R, Z, Zi, A]
    df_algs = pd.DataFrame(alg).T
    # df_algs.replace(0.0, 500.0, inplace=True)
    df_algs.columns = ['C', 'R', 'Z', 'Zi', 'AVG']
    # df_objs.columns = ['obj1', 'obj7']
    df_algs.index = ['alg1', 'alg2', 'alg3', 'alg4', 'alg5', 'alg6', 'alg7', 'alg8']
    # df_objs.index = ['alg2']
    plot = df_algs.plot.line(colormap='rainbow', figsize=(10,10),linewidth=1) #  logy=True,
    plot.set(xlabel='algorithm', ylabel='sharing rate')
    fig = plot.get_figure()
    # fig_path = os.path.join(BASE_DIR, 'results', 'CRZ', ty)
    fig_path = os.path.join(BASE_DIR, 'results', no + 'req_analysis')
    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(fig_path + '/' + no + 'req_sharing_rates.png')
    # fig.savefig(fig_path + '/'  + 'C.png')
    res_name = no + 'req_sharing_rates.csv'
    # res_name = 'C.csv'
    pathout = os.path.join(fig_path, res_name)
    df_algs.to_csv(pathout)


def sharing():
    no = '100'
    folder = os.path.join(BASE_DIR, 'results', no + 'req')
    alg = []
    for resultFile in glob.iglob(os.path.join(folder, '*.csv')):
        filename = os.path.splitext(os.path.basename(resultFile))[0]
        df = pd.read_csv(resultFile)
        if '-C' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'C', df.ix[0, 'avg_sharing']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'C', df.ix[0, 'avg_sharing']])
        elif '-R' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'R', df.ix[0, 'avg_sharing']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'R', df.ix[0, 'avg_sharing']])
        elif '_i_' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'Zi', df.ix[0, 'avg_sharing']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'Zi', df.ix[0, 'avg_sharing']])
        elif '-Z' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'Z', df.ix[0, 'avg_sharing']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'Z', df.ix[0, 'avg_sharing']])
    df_algs = pd.DataFrame(alg)
    # df_algs.replace(0.0, 500.0, inplace=True)
    df_algs.columns = ['algorithm', 'data_type', 'sharing_rate']
    ax = sns.catplot(data=df_algs, x='algorithm', y='sharing_rate', hue='data_type', kind='violin')
    # ax1 = sns.violinplot()
    # plt.set(xlabel='algorithm', ylabel='sharing rate')
    # fig = ax.get_figure()
    # fig_path = os.path.join(BASE_DIR, 'results', 'CRZ', ty)
    fig_path = os.path.join(BASE_DIR, 'results', no + 'req_analysis')
    os.makedirs(fig_path, exist_ok=True)
    ax.savefig(fig_path + '/' + no + 'req_sharing_rates_bar.png')
    # fig.savefig(fig_path + '/'  + 'C.png')
    res_name = no + 'req_sharing_rates.csv'
    # res_name = 'C.csv'
    pathout = os.path.join(fig_path, res_name)
    df_algs.to_csv(pathout)


def plot():
    no = '100'
    folder = os.path.join(BASE_DIR, 'results', no + 'req_analysis')
    for resultFile in glob.iglob(os.path.join(folder, '*.csv')):
        # filename = os.path.splitext(os.path.basename(resultFile))[0]
        df = pd.read_csv(resultFile)
        ax = sns.catplot(data=df, col='algorithm', y='sharing_rate', hue='data_type', kind='violin')
        fig_path = os.path.join(BASE_DIR, 'results', no + 'req_analysis')
        os.makedirs(fig_path, exist_ok=True)
        ax.savefig(fig_path + '/' + no + 'req_sharing_rates_violin.png')


def aggregate():
    no = '1000'
    folder = os.path.join(BASE_DIR, 'results', no + 'req')
    alg = []
    for resultFile in glob.iglob(os.path.join(folder, '*.csv')):
        filename = os.path.splitext(os.path.basename(resultFile))[0]
        df = pd.read_csv(resultFile)
        if '-C' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
        elif '-R' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
        elif '_i_' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
        elif '-Z' in filename:
            if 'alg1' in filename:
                alg.append(['alg1', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append(['alg2', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append(['alg3', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append(['alg4', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append(['alg5', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append(['alg6', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append(['alg7', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append(['alg8', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
    df_algs = pd.DataFrame(alg)
    df_algs.columns = ['algorithm', 'data_type', 'sharing_rate', 'avg_req', 'avg_dist', 'avg_cross', 'num_veh', 'min_fitness']
    # ax = sns.catplot(data=df_algs, x='algorithm', y='sharing_rate', hue='data_type', kind='violin')
    # ax1 = sns.violinplot()
    # plt.set(xlabel='algorithm', ylabel='sharing rate')
    # fig = ax.get_figure()
    # fig_path = os.path.join(BASE_DIR, 'results', 'CRZ', ty)
    fig_path = os.path.join(BASE_DIR, 'results', 'analysis')
    os.makedirs(fig_path, exist_ok=True)
    # ax.savefig(fig_path + '/' + no + 'req_sharing_rates_bar.png')
    # fig.savefig(fig_path + '/'  + 'C.png')
    res_name = no + 'req.csv'
    # res_name = 'C.csv'
    pathout = os.path.join(fig_path, res_name)
    df_algs.to_csv(pathout)



def dmd():
    folder = os.path.join(BASE_DIR, 'results', '100req0', 'd')
    alg = []
    for resultFile in glob.iglob(os.path.join(folder, '*.csv')):
        filename = os.path.splitext(os.path.basename(resultFile))[0]
        df = pd.read_csv(resultFile)
        if 'dmdA' in filename:
            demand = 'equal_dmd'
        elif 'dmd1' in filename:
            demand = 'dmd1'
        elif 'dmd2' in filename:
            demand = 'dmd2'
        elif 'dmd3' in filename:
            demand = 'dmd3'
        if '-C' in filename:
            if 'alg1' in filename:
                alg.append([demand, 'alg1', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append([demand, 'alg2', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append([demand, 'alg3', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append([demand, 'alg4', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append([demand, 'alg5', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append([demand, 'alg6', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append([demand, 'alg7', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append([demand, 'alg8', 'C', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
        elif '-R' in filename:
            if 'alg1' in filename:
                alg.append([demand, 'alg1', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append([demand, 'alg2', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append([demand, 'alg3', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append([demand, 'alg4', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append([demand, 'alg5', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append([demand, 'alg6', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append([demand, 'alg7', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append([demand, 'alg8', 'R', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
        elif '_i_' in filename:
            if 'alg1' in filename:
                alg.append([demand, 'alg1', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append([demand, 'alg2', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append([demand, 'alg3', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append([demand, 'alg4', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append([demand, 'alg5', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append([demand, 'alg6', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append([demand, 'alg7', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append([demand, 'alg8', 'Zi', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
        elif '-Z' in filename:
            if 'alg1' in filename:
                alg.append([demand, 'alg1', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg2' in filename:
                alg.append([demand, 'alg2', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg3' in filename:
                alg.append([demand, 'alg3', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg4' in filename:
                alg.append([demand, 'alg4', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg5' in filename:
                alg.append([demand, 'alg5', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg6' in filename:
                alg.append([demand, 'alg6', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg7' in filename:
                alg.append([demand, 'alg7', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
            elif 'alg8' in filename:
                alg.append([demand, 'alg8', 'Z', df.ix[0, 'avg_sharing'], df.ix[0, 'avg_req'], df.ix[0, 'avg_dist'], df.ix[0, 'avg_cross'], df.ix[0, 'num_veh'], df.ix[0, 'min_fitness']])
    df_algs = pd.DataFrame(alg)
    df_algs.columns = ['demand', 'algorithm', 'data_type', 'sharing_rate', 'avg_req', 'avg_dist', 'avg_cross', 'num_veh', 'min_fitness']
    # ax = sns.catplot(data=df_algs, x='algorithm', y='sharing_rate', hue='data_type', kind='violin')
    # ax1 = sns.violinplot()
    # plt.set(xlabel='algorithm', ylabel='sharing rate')
    # fig = ax.get_figure()
    # fig_path = os.path.join(BASE_DIR, 'results', 'CRZ', ty)
    fig_path = os.path.join(BASE_DIR, 'results', 'analysis')
    os.makedirs(fig_path, exist_ok=True)
    # ax.savefig(fig_path + '/' + no + 'req_sharing_rates_bar.png')
    # fig.savefig(fig_path + '/'  + 'C.png')
    res_name = '100req_demand.csv'
    # res_name = 'C.csv'
    pathout = os.path.join(fig_path, res_name)
    df_algs.to_csv(pathout)


# sharing()
# plot()
# aggregate()
dmd()
