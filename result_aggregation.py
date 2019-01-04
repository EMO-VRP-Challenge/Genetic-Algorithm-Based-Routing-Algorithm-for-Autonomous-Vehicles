# -*- coding: utf-8 -*-
# result_aggregation.py

import csv
import fnmatch
import glob
import os
import re
import numpy as np
from matplotlib import cm

from src import utils, BASE_DIR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from io import StringIO
# import prettytable

def all_gen_aggregation():
    csvPathname1 = os.path.join(BASE_DIR, 'results', 'analysis', 'p1100m1')
    csvFilename2 = 'all_pop_m1_G150.csv'
    csvPathname2 = os.path.join(BASE_DIR, 'resultAnalysis', 'pop1000', 'cro', csvFilename2)
    column = []
    for f in [os.path.join(csvPathname1, csvFilename) for csvFilename in fnmatch.filter(os.listdir(csvPathname1), '*csv')]:
        df = pd.read_csv(f)
        df = df.drop(df.index[len(df)-1])
        x = df.columns[2]
        y = df[x]
        z = pd.DataFrame(y)
        # column.append(pd.DataFrame(df[df.columns[2]]))
        column.append(z)
        # column.append(pd.DataFrame(df[df.iloc[-2, 2]]))
    merged_dataframe = pd.concat(column, axis=1)
    # print(merged_dataframe)
    merged_dataframe.to_csv(csvPathname2, index=True)


def final_fit_aggregation():
    for i in ['1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000']:
        for j in range(1,11):
            name = 'p' + i + 'm' + str(j)
            combine_name = name + '.csv'
            csvPathname1 = os.path.join(BASE_DIR, 'results', 'analysis', name)
            csvFilename2 = combine_name
            csvPathname2 = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'pop', csvFilename2)
            final_fit = []
            for f in [os.path.join(csvPathname1, csvFilename) for csvFilename in fnmatch.filter(os.listdir(csvPathname1), '*csv')]:
                df = pd.read_csv(f)
                # df = df.drop(df.index[len(df)-1])
                x = df.iloc[-2, 2]
                # y = df[x]
                y = pd.DataFrame([x])
                # final_fit.append(pd.DataFrame(df[df.iloc[-2, 2]]))
                final_fit.append(y)
            merged_dataframe = pd.concat(final_fit)
            # print(merged_dataframe)
            merged_dataframe.to_csv(csvPathname2, index=False)


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def combine_final_fit():
    csvPathname1 = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'pop')
    csvFilename2 = 'combined_pop_G150.csv'
    csvPathname2 = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', csvFilename2)
    column = []
    for f in sorted(glob.glob(os.path.join(csvPathname1, '*.csv')), key=numericalSort):
    # for f in [os.path.join(csvPathname1, csvFilename) for csvFilename in fnmatch.filter(os.listdir(csvPathname1), '*csv')]:
        df = pd.read_csv(f)
        # df = df.drop(df.index[0])
        z = pd.DataFrame(df[df.columns[0]])
        # column.append(pd.DataFrame(df[df.columns[2]]))
        column.append(z)
        # column.append(pd.DataFrame(df[df.iloc[-2, 2]]))
    merged_dataframe = pd.concat(column, axis=0)
    # print(merged_dataframe)
    merged_dataframe.to_csv(csvPathname2, index=False)


def aggregation_plot():
    x = 20
    cx = [0.8, 0.7, 0.9, 0.3, 0.9, 0.8]
    mut = [0.4, 0.3, 0.7, 0.2, 0.8, 0.7]
    plt.style.use('ggplot')
    path = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'repeat')
    # for f in [os.path.join(csvPathname3, csvFilename) for csvFilename in fnmatch.filter(os.listdir(csvPathname3), '*csv')]:
    for c, m in zip(cx, mut):
        file_name0 = 'cx' + str(c) + 'mut' + str(m)
        file_name = file_name0 + '.csv'
        path_in = os.path.join(path, file_name)
        df = pd.read_csv(path_in)
        # df.drop(df.columns[[0]], axis=1, inplace=True)
        df = df[x:]
        plt.figure()
        ax = df.plot(figsize=(20,10), title=file_name0, colormap='rainbow')#  gist_rainbow
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')

        # plt.show()
        # Save to folder
        # filename = os.path.splitext(f)[0]
        # path = os.path.dirname(os.path.abspath(path_in))
        file_name0 = file_name0 + '_start' + str(x)
        path_out = os.path.join(path, file_name0) #'\cx0.8mut0.4_start' + str(x)
        # print(filename)
        plt.savefig(path_out + '.png')


def box_plot():
    path_in = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'pop', 'G150_p1000.csv')
    path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'pop', 'G150_p1000.png') # _with_hue_GnBu_d , palette='GnBu_d'
    data = pd.read_csv(path_in)
    ax = sns.boxplot(x="population", y="fitness", data=data, palette='coolwarm') #
    # ax = sns.swarmplot(x="crossover", y="fitness", hue="population", benchmark=benchmark, palette='GnBu_d') #
    # ax = sns.regplot(x="crossover", y="fitness", benchmark=benchmark)
    fig = ax.get_figure()
    fig.savefig(path_out)


def line_plot(population):
    path_in = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'G150.csv')#'pop', 'G150_p1000.csv')
    path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot')

    data = pd.read_csv(path_in)
    data_index = data.set_index(['population', 'mutation', 'crossover']).sort_index()
    # print(data_index)

    # color_idx = np.linspace(1, 0, 10)
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    for i, m in zip(colors, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        df = data_index.loc[population, m]
        # df = data_index[population, mutation, 0.8]
        # print(df)
        plt.plot(df.index, df.values, label="mutation={0}".format(m), color=i)
        plt.xlabel('Crossover')
        plt.ylabel('Fitness')
        title = 'Population: ' + str(population)
        plt.title(title)
        plt.legend()

    fig_name = 'G150_p' + str(population) + '.png'
    path_name = os.path.join(path_out, fig_name)
    plt.savefig(path_name)


def line_plot_mut(population):
    path_in = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'G150.csv')#'pop', 'G150_p1000.csv')
    path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot')

    data = pd.read_csv(path_in)
    data_index = data.set_index(['population', 'crossover', 'mutation']).sort_index()
    # print(data_index)

    # color_idx = np.linspace(1, 0, 10)
    cmap = plt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    for i, m in zip(colors, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        df = data_index.loc[population, m]
        # df = data_index[population, mutation, 0.8]
        # print(df)
        plt.plot(df.index, df.values, label="crossover={0}".format(m), color=i)
        plt.xlabel('Mutation')
        plt.ylabel('Fitness')
        title = 'Population: ' + str(population)
        plt.title(title)
        plt.legend()

    fig_name = 'G150_p' + str(population) + '_m.png'
    path_name = os.path.join(path_out, fig_name)
    plt.savefig(path_name)


def line_plot_pop():
    cx = [0.8, 0.7, 0.9, 0.3, 0.9, 0.8]
    mut = [0.4, 0.3, 0.7, 0.2, 0.8, 0.7]
    path = os.path.join(BASE_DIR, 'results', 'repeat')
    for c, m in zip(cx, mut):
        column = []
        file_name = 'cx' + str(c) + 'mut' + str(m) + '.csv'
        path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'repeat', file_name)
        for pop in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
            file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS' + str(pop) + '_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
            path_in = os.path.join(path, file)
            df = pd.read_csv(path_in)
            df = df.drop(df.index[len(df)-1])
            sub_df = pd.DataFrame(df[df.columns[2]])
            column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=1)
        # print(merged_dataframe)
        merged_dataframe.to_csv(path_out, index=True)


def select_best_CxMt():
    path_in = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'G150.csv')
    path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'Best_fit_comb_c_LargerThan_m.csv')
    data = pd.read_csv(path_in)
    # data1000 = benchmark[benchmark.population == 2000]
    min_fit = data[data.fitness < 0.0002]
    # min_fit = min_fit[benchmark.crossover > benchmark.mutation]
    # min_fit = min_fit[benchmark.crossover > 0.1]
    # min_fit = min_fit[benchmark.mutation > 0.1]
    # min_fit = min_fit[benchmark.crossover < 1.0]
    # min_fit = min_fit[benchmark.mutation < 1.0]
    min_fit = min_fit[data.crossover == 0.9]
    min_fit = min_fit[data.mutation == 0.8]
    # print(min_fit)
    min_fit.to_csv(path_out, index=True)


def group_by_pop():
    cx = [0.8, 0.7, 0.9, 0.3, 0.9, 0.8]
    mut = [0.4, 0.3, 0.7, 0.2, 0.8, 0.7]
    path = os.path.join(BASE_DIR, 'results', 'repeat')
    for pop in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
        column = []
        file_name = 'pop' + str(pop) + '.csv'
        path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'Plot', 'repeat', file_name)
        for c, m in zip(cx, mut):
            file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS' + str(pop) + '_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
            path_in = os.path.join(path, file)
            df = pd.read_csv(path_in)
            new_columns = df.columns.values
            new_columns[2] = 'fitness' # give new column name to the selected column
            df.columns = new_columns
            df = df.drop(df.index[len(df)-1])
            # sub_df = pd.DataFrame(df[df.columns[0,2]])
            # df = df.iloc[:, [0, 2]]
            sub_df = pd.DataFrame(df[df.columns[2]]) # write the new constructed dataframe
            new_column = 'c' + str(c) + 'm' + str(m)
            # output = StringIO()
            # sub_df.to_csv(output)
            # output.seek(0)
            # pt = prettytable.from_csv(output)
            # print(pt)
            # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
            #     print(sub_df)
            sub_df['CxMut'] = new_column # add a new column to specify their respective cx, mut values
            sub_df['population'] = pop
            sub_df['generation'] = [i for i in range(150)]
            # print(sub_df.to_string())
            column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # print(merged_dataframe)
        merged_dataframe.to_csv(path_out, index=False)


def makeDirsForFile(pathname):
    try:
        os.makedirs(pathname, exist_ok=True) # os.path.split(pathname)[0]
    except:
        pass


# aggregate the min fitness from different cx, mut operators results
# def S_CM_operators():
#     # for operator in ['OrdShu', 'OrdInv', 'PmInv', 'PmShu']: #
#     aggr_list = []
#     for cro in ['Pm', 'Ord']: #
#         for mut in ['Shu', 'Inv']:
#             operator = cro + mut
#             path_i = os.path.join(BASE_DIR, 'results', 'Solomon', operator)#, folder)
#             dirs = [d for d in os.listdir(path_i) if os.path.isdir(os.path.join(path_i, d))] # a list of folder names (strings)
#             path = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'CM_operators')
#             os.makedirs(path, exist_ok=True)
#             file_out = operator + '.csv'
#             path_out = os.path.join(path, file_out3)
#             path_out2 = os.path.join(path, 'S_oper.csv')
#             path_out3 = os.path.join(path, 'S_oper_stats.csv')
#             min_row = []
#
#                 # file_out = folder + '.csv'
#                 # path_out2 = os.path.join(path_out, file_out)
#                 # makeDirsForFile(pathname=path_out2)
#             column = []
#             for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#                 for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#                     for folder in dirs:
#                         path_in = os.path.join(path_i, folder)
#
#                         file = str(folder) + '_cro' + cro + '_mut' + mut + '_selTour_wC100.0_dC100.0_iS100_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
#                         path2 = os.path.join(path_in, file)
#                         df = pd.read_csv(path2)
#                         sub_df = pd.DataFrame([df.iloc[-2, 2]]) #pd.DataFrame(df[df.columns[2]])
#                         sub_df.columns = ['fitness']
#                         new_column = 'c' + str(c) + 'm' + str(m)
#                         sub_df['CxMut'] = new_column
#                         column.append(sub_df)
#             merged_dataframe = pd.concat(column, axis=0)
#             merged_dataframe.reset_index(drop=True, inplace=True)
#             # ind = merged_dataframe['fitness'].idxmin(axis=1)
#             # df_min = merged_dataframe.iloc[[ind]]
#             avg = merged_dataframe['fitness'].mean()
#             medi = merged_dataframe['fitness'].median()
#             df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
#             df_min['avg'] = avg
#             df_min['median'] = medi
#             df_min['operator'] = operator
#             # df_min['instance'] = folder
#             min_row.append(df_min)
#                 # merged_dataframe.to_csv(path_out2, index=False)
#             merged_min_df = pd.concat(min_row, axis=0)
#             aggr_df = pd.DataFrame([merged_min_df.iloc[0:]])
#             aggr_list.append(aggr_df)
#             merged_min_df.to_csv(path_out, index=False)
#     merged_aggr_df = pd.concat(aggr_list, axis=0)
#     merged_aggr_df.to_csv(path_out2, index=False)



def S_CM_operators():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'CM_operators')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'S_oper_all_oper.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'S_oper_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for cro in ['Ord', 'Pm']: #
        for mut in ['Inv', 'Shu']: #
            operator = cro + mut
            file_out = operator + '.csv'
            path_out = os.path.join(dir_out, file_out)
            path = os.path.join(BASE_DIR, 'results', 'Solomon', 'oper', operator)
            dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            column = []
            for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    for folder in dirs:
                        file = str(folder) + '_cro' + cro + '_mut' + mut + '_selTour_wC100.0_dC100.0_iS100_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                        path_i = os.path.join(path, folder)
                        path_in = os.path.join(path_i, file)
                        df = pd.read_csv(path_in)
                        sub_df = pd.DataFrame([df.iloc[-2, 2]])
                        sub_df.columns = ['fitness']
                        c_column = str(c)
                        sub_df['Cx'] = c_column
                        m_column = str(m)
                        sub_df['Mut'] = m_column
                        sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                        if folder[0] == 'C':
                            sub_df['inst'] = 11
                        elif folder[:2] == 'RC':
                            sub_df['inst'] = 33
                        else:
                            sub_df['inst'] = 22
                        column.append(sub_df)
            merged_dataframe = pd.concat(column, axis=0)
            # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
            df_min = merged_dataframe.nsmallest(18144, columns=['fitness'])
            # df_min['oper'] = operator
            if operator == 'OrdInv':
                df_min['oper'] = 1
            elif operator == 'OrdShu':
                df_min['oper'] = 2
            elif operator == 'PmInv':
                df_min['oper'] = 3
            else:
                df_min['oper'] = 4
            avg = merged_dataframe['fitness'].mean()
            medi = merged_dataframe['fitness'].median()
            df_min['F_avg'] = avg
            df_min['F_medi'] = medi
            df_min['T_avg'] = merged_dataframe['time'].mean()
            df_min['T_medi'] = merged_dataframe['time'].mean()
            aggr_list.append(df_min)
            # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False

    stats_df = merged_min_df.groupby('oper', as_index=False)['time'].median()
    stats_df.columns = ['oper','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('oper', as_index=False)['fitness'].median().iloc[:,1]
    stats_df['T_avg'] =merged_min_df.groupby('oper' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('oper', as_index=False)['fitness'].mean().iloc[:,1]
    stats_df['T_min'] =merged_min_df.groupby('oper' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('oper', as_index=False)['fitness'].min().iloc[:,1]

    # stats_df = merged_min_df.groupby(['inst', 'oper'], as_index=False)['time'].median()
    # stats_df.columns = ['inst', 'oper', 'T_medi']
    # stats_df['fit_medi']=merged_min_df.groupby(['oper', 'inst'], as_index=False)['fitness'].median().iloc[:,2]
    # # stats_df['C_medi']= merged_min_df.groupby(['oper', 'inst'], as_index=False)['Cx'].median().iloc[:,1]   #
    # # stats_df['M_medi']=merged_min_df.groupby(['oper', 'inst'], as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    # stats_df['T_avg'] =merged_min_df.groupby(['oper', 'inst'] , as_index=False)['time'].mean().iloc[:,2] # ['M_medi']
    # stats_df['fit_avg']=merged_min_df.groupby(['oper', 'inst'], as_index=False)['fitness'].mean().iloc[:,2]
    # # stats_df['C_avg']  = merged_min_df.groupby('oper' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # # stats_df['M_avg'] = merged_min_df.groupby('oper' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    # stats_df['T_min'] =merged_min_df.groupby(['oper', 'inst'] , as_index=False)['time'].min().iloc[:,2]
    # stats_df['fit_min']=merged_min_df.groupby(['oper', 'inst'], as_index=False)['fitness'].min().iloc[:,2]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':2, 'fit_avg':2, 'T_min':3, 'fit_min':7})
    stats_df.to_csv(path_out3, index=False)




def A_CM_operators():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'CM_operators')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'A_oper_min.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'A_oper_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for cro in ['Ord', 'Pm']: #
        for mut in ['Inv', 'Shu']: #
            operator = cro + mut
            file_out = operator + '.csv'
            path_out = os.path.join(dir_out, file_out)
            path = os.path.join(BASE_DIR, 'results', 'A', 'operator', operator)
            column = []
            for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    file = 'A-n16-k4_cro' + cro + '_mut' + mut + '_selTour_wC100.0_dC100.0_iS16_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                    path_in = os.path.join(path, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    sub_df['Cx'] = c_column
                    m_column = str(m)
                    sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    column.append(sub_df)
            merged_dataframe = pd.concat(column, axis=0)
            # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
            df_min = merged_dataframe.nsmallest(324, columns=['fitness'])
            # df_min['oper'] = operator
            if operator == 'OrdInv':
                df_min['oper'] = 1
            elif operator == 'OrdShu':
                df_min['oper'] = 2
            elif operator == 'PmInv':
                df_min['oper'] = 3
            else:
                df_min['oper'] = 4
            avg = merged_dataframe['fitness'].mean()
            medi = merged_dataframe['fitness'].median()
            df_min['F_avg'] = avg
            df_min['F_medi'] = medi
            df_min['T_avg'] = merged_dataframe['time'].mean()
            df_min['T_medi'] = merged_dataframe['time'].mean()
            aggr_list.append(df_min)
            # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('oper', as_index=False)['time'].median()
    stats_df.columns = ['oper','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('oper', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('oper', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('oper', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('oper' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('oper', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('oper' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('oper' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('oper' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('oper', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'T_min':3, 'fit_min':7})
    stats_df.to_csv(path_out3, index=False)


# def A_CM_operators():
#     # for operator in ['OrdShu', 'OrdInv', 'PmInv', 'PmShu']: #
#     aggr_list = []
#     for cro in ['Pm', 'Ord']: #
#         for mut in ['Shu', 'Inv']:
#             operator = cro + mut
#             path_in = os.path.join(BASE_DIR, 'results', 'A', 'operator', operator)#, folder)
#             # dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))] # a list of folder names (strings)
#             path_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'CM_operators', operator)
#             os.makedirs(path_out, exist_ok=True)
#             path_out3 = os.path.join(path_out, '5mins_fitness.csv')
#             path_out4 = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'CM_operators', 'A_5averages.csv')
#             min_row = []
#             # for folder in dirs:
#             #     path_in = os.path.join(path, folder)
#             file_out = 'A.csv'
#             path_out2 = os.path.join(path_out, file_out)
#                 # makeDirsForFile(pathname=path_out2)
#             column = []
#             for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
#                 for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
#                     file = 'A-n16-k4_cro' + cro + '_mut' + mut + '_selTour_wC100.0_dC100.0_iS16_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv' # str(folder) +
#                     path2 = os.path.join(path_in, file)
#                     df = pd.read_csv(path2)
#                     # new_columns = df.columns.values
#                     # new_columns[2] = 'fitness'
#                     # df.columns = new_columns
#                     # df = df.drop(df.index[len(df)-1])
#                     sub_df = pd.DataFrame([df.iloc[-2, 2]]) #pd.DataFrame(df[df.columns[2]])
#                     sub_df.columns = ['fitness']
#                     new_column = 'c' + str(c) + 'm' + str(m)
#                     sub_df['CxMut'] = new_column
#                     column.append(sub_df)
#             merged_dataframe = pd.concat(column, axis=0)
#             # merged_dataframe.reset_index(drop=True, inplace=True)
#             # ind = merged_dataframe['fitness'].idxmin(axis=1)
#             # df_min = merged_dataframe.iloc[[ind]]
#             avg = merged_dataframe['fitness'].mean()
#             medi = merged_dataframe['fitness'].median()
#             # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
#             # df_min = merged_dataframe[merged_dataframe.fitness < 0.004]
#             df_min = merged_dataframe.nsmallest(20, columns=['fitness'])
#             df_min['avg'] = avg
#             df_min['median'] = medi
#             df_min['operator'] = operator
#             # df_min['instance'] = folder
#             min_row.append(df_min)
#             # merged_dataframe.to_csv(path_out2, index=False)
#             merged_min_df = pd.concat(min_row, axis=0)
#             aggr_df = pd.DataFrame([merged_min_df.iloc[0:]])
#             aggr_list.append(aggr_df)
#             merged_min_df.to_csv(path_out3, index=False)
#     merged_aggr_df = pd.concat(aggr_list, axis=0)
#     merged_aggr_df.to_csv(path_out4, index=False)


# A_pop
def A_20sets_aggregation():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'pop')
    # os.makedirs(dir_out, exist_ok=True)
    file_out3 = 'A_aggr20.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    file_out4 = 'A_stats20_all.csv'
    path_out4 = os.path.join(dir_out, file_out4)
    aggr_list = []
    stats_list = []
    for p in ['1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000']:
        file_out2 = 'p' + p + '.csv'
        path_out2 = os.path.join(dir_out, file_out2)
        column = []
        for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
            for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
                file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS' + str(p) + '_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                for setno in range(1, 21):
                    set = 'set' + str(setno)
                    path = os.path.join(BASE_DIR, 'results', 'A-n16-k4_20sets', set)
                    # file_out = set + '.csv'
                    # path_out = os.path.join(dir_out, file_out)
                    path_in = os.path.join(path, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    sub_df['Cx'] = c_column
                    m_column = str(m)
                    sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(17820, columns=['fitness'])
        df_min['pop'] = p
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['Favg'] = avg
        df_min['Fmedi'] = medi
        df_min['Tavg'] = merged_dataframe['time'].mean()
        df_min['Tmedi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out2, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out3, index=False)
    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('pop', as_index=False)['time'].median()
    stats_df.columns = ['pop','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('pop', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('pop', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('pop', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('pop' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('pop', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('pop' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('pop' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('pop' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('pop', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7})
    # concat_stats_df = pd.concat([stats_df, stats_df2, stats_df3, stats_df4, stats_df5, stats_df6], axis=1, keys=('T_medi', 'C_medi', 'M_medi', 'T_avg', 'C_avg', 'M_avg'))
    # stats_list.append(stats_df)
    # merged_stats_df = pd.concat(stats_list, axis=0)
    stats_df.to_csv(path_out4, index=False)


def A_selection():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'selection')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'A_sel.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'A_sel_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for sel in ['Rou', 'Tour']:
        file_out = sel + '.csv'
        path_out = os.path.join(dir_out, file_out)
        path = os.path.join(BASE_DIR, 'results', 'A', 'selection', sel)
        column = []
        for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
            for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
                file = 'A-n16-k4_croOrd_mutShu_sel' + sel + '_wC100.0_dC100.0_iS16_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                path_in = os.path.join(path, file)
                df = pd.read_csv(path_in)
                sub_df = pd.DataFrame([df.iloc[-2, 2]])
                sub_df.columns = ['fitness']
                c_column = str(c)
                sub_df['Cx'] = c_column
                m_column = str(m)
                sub_df['Mut'] = m_column
                sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(162, columns=['fitness'])
        # df_min['sel'] = sel
        if sel == 'Rou':
            df_min['sel'] = '1'
        else:
            df_min['sel'] = '2'
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('sel', as_index=False)['time'].median()
    stats_df.columns = ['sel','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('sel', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('sel', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('sel', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('sel' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('sel', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('sel' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('sel' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('sel' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('sel', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':4, 'fit_avg':4, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'sel':0, 'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7})
    stats_df.to_csv(path_out3, index=False)



def A_tour_size():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'tourSize')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'A_tourSize.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'A_tourSize_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for no in ['1', '2', '3']:
        file_out = no + '.csv'
        path_out = os.path.join(dir_out, file_out)
        path = os.path.join(BASE_DIR, 'results', 'A', 'tourSize', no)
        column = []
        for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
            for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
                file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                path_in = os.path.join(path, file)
                df = pd.read_csv(path_in)
                sub_df = pd.DataFrame([df.iloc[-2, 2]])
                sub_df.columns = ['fitness']
                c_column = str(c)
                sub_df['Cx'] = c_column
                m_column = str(m)
                sub_df['Mut'] = m_column
                sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(243, columns=['fitness'])
        df_min['tourSize'] = no
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('tourSize', as_index=False)['time'].median()
    stats_df.columns = ['tourSize','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('tourSize', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('tourSize', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('tourSize', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('tourSize' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('tourSize', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('tourSize' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('tourSize' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('tourSize' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('tourSize', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':4, 'fit_avg':4, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'fitness':7, 'time':3})
    stats_df.to_csv(path_out3, index=False)


def S_pop():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'pop')
    os.makedirs(dir_out, exist_ok=True)
    file_out3 = 'S_pop.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    file_out4 = 'S_pop_stats_all.csv'
    path_out4 = os.path.join(dir_out, file_out4)
    aggr_list = []
    stats_list = []
    path = os.path.join(BASE_DIR, 'results', 'Solomon', 'pop')
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for p in ['100', '500', '1000', '1500']:
        file_out2 = 'p' + p + '.csv'
        path_out2 = os.path.join(dir_out, file_out2)
        # file_out = folder + '.csv'
        # path_out = os.path.join(dir_out, file_out)
        column = []
        for m in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            for c in [0.6, 0.7, 0.8]: #0.1, 0.2, 0.3, 0.4, 0.5, , 0.9
                for folder in dirs:
                    file = folder + '_croOrd_mutShu_selTour_wC100.0_dC100.0_iS100_pS' + p + '_cP' + str(c) + '_mP' + str(m) + '_nG200.csv'
                    path_i = os.path.join(path, folder)
                    path_in = os.path.join(path_i, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    sub_df['Cx'] = c_column
                    m_column = str(m)
                    sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    # sub_df['inst'] = folder
                    column.append(sub_df)
            merged_dataframe = pd.concat(column, axis=0)
            # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
            df_min = merged_dataframe.nsmallest(36, columns=['fitness'])
            df_min['pop'] = p
            avg = merged_dataframe['fitness'].mean()
            medi = merged_dataframe['fitness'].median()
            df_min['F_avg'] = avg
            df_min['F_medi'] = medi
            df_min['T_avg'] = merged_dataframe['time'].mean()
            df_min['T_medi'] = merged_dataframe['time'].mean()
            aggr_list.append(df_min)
            # merged_dataframe.to_csv(path_out2, index=False)
        merged_min_df = pd.concat(aggr_list, axis=0)
        # merged_min_df.to_csv(path_out3, index=False)

        merged_min_df = merged_min_df.astype(float) # , copy=False
        stats_df = merged_min_df.groupby('pop', as_index=False)['time'].median()
        stats_df.columns = ['pop','T_medi']
        stats_df['fit_medi']=merged_min_df.groupby('pop', as_index=False)['fitness'].median().iloc[:,1]
        # stats_df['C_medi']= merged_min_df.groupby('pop', as_index=False)['Cx'].median().iloc[:,1]   #
        # stats_df['M_medi']=merged_min_df.groupby('pop', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
        stats_df['T_avg'] =merged_min_df.groupby('pop' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
        stats_df['fit_avg']=merged_min_df.groupby('pop', as_index=False)['fitness'].mean().iloc[:,1]
        # stats_df['C_avg']  = merged_min_df.groupby('pop' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
        # stats_df['M_avg'] = merged_min_df.groupby('pop' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
        stats_df['T_min'] =merged_min_df.groupby('pop' , as_index=False)['time'].min().iloc[:,1]
        stats_df['fit_min']=merged_min_df.groupby('pop', as_index=False)['fitness'].min().iloc[:,1]
        stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':2, 'fit_avg':2, 'T_min':3, 'fit_min':7})
        # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':4, 'fit_avg':4, 'fitness':7, 'time':3})
        stats_df.to_csv(path_out4, index=False)


def S_selection():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'selection')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'S_sel.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'S_sel_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for sel in ['Rou', 'Tour']:
        file_out = sel + '.csv'
        path_out = os.path.join(dir_out, file_out)
        path = os.path.join(BASE_DIR, 'results', 'Solomon', sel)
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        column = []
        for m in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            for c in [0.6, 0.7, 0.8]:
                for folder in dirs:
                    file = folder + '_croOrd_mutShu_sel' + sel + '_wC100.0_dC100.0_iS100_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                    path_i = os.path.join(path, folder)
                    path_in = os.path.join(path_i, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    # sub_df['Cx'] = c_column
                    m_column = str(m)
                    # sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    # sub_df['inst'] = folder
                    column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(54, columns=['fitness'])
        if sel == 'Rou':
            df_min['sel'] = '1'
        else:
            df_min['sel'] = '2'
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df = merged_min_df.round({'fitness':6, 'T_medi':3, 'T_avg':3, 'F_medi':3, 'F_avg':3, 'time':3})
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('sel', as_index=False)['time'].median()
    stats_df.columns = ['sel','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('sel', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('sel', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('sel', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('sel' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('sel', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('sel' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('sel' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('sel' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('sel', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':2, 'T_avg':2, 'fit_medi':2, 'fit_avg':2, 'T_min':3, 'fit_min':5})
    # stats_df = stats_df.round({'sel':0, 'T_medi':3, 'T_avg':3, 'fit_medi':4, 'fit_avg':4, 'M_avg':2})
    stats_df.to_csv(path_out3, index=False)


def S_tourSize():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'tourSize')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'S_tourSize.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'S_tourSize_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for no in ['1', '2', '3']:
        file_out = no + '.csv'
        path_out = os.path.join(dir_out, file_out)
        path = os.path.join(BASE_DIR, 'results', 'Solomon', 'tourSize', no)
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        column = []
        for m in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            for c in [0.6, 0.7, 0.8]:
                for folder in dirs:
                    file = folder + '_croOrd_mutShu_selTour_wC100.0_dC100.0_iS100_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG150.csv'
                    path_i = os.path.join(path, folder)
                    path_in = os.path.join(path_i, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    sub_df['Cx'] = c_column
                    m_column = str(m)
                    sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    # sub_df['inst'] = folder
                    column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(27, columns=['fitness'])
        df_min['tourSize'] = no
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('tourSize', as_index=False)['time'].median()
    stats_df.columns = ['tourSize','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('tourSize', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('tourSize', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('tourSize', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('tourSize' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('tourSize', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('tourSize' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('tourSize' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('tourSize' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('tourSize', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':4, 'fit_avg':4, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'fitness':7, 'time':3})
    stats_df.to_csv(path_out3, index=False)



def A_gen():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'gen')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'A_gen.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'A_gen_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for g in ['50', '100', '150', '200', '250']:
        file_out = 'g' + g + '.csv'
        path_out = os.path.join(dir_out, file_out)
        folder_in = 'g' + g
        path = os.path.join(BASE_DIR, 'results', 'A', 'gen', folder_in)
        column = []
        for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
            for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
                file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS1000_cP' + str(c) + '_mP' + str(m) + '_nG' + g + '.csv'
                path_in = os.path.join(path, file)
                df = pd.read_csv(path_in)
                sub_df = pd.DataFrame([df.iloc[-2, 2]])
                sub_df.columns = ['fitness']
                c_column = str(c)
                sub_df['Cx'] = c_column
                m_column = str(m)
                sub_df['Mut'] = m_column
                sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(405, columns=['fitness'])
        df_min['gen'] = g
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('gen', as_index=False)['time'].median()
    stats_df.columns = ['gen','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('gen', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('gen', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('gen', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('gen' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('gen', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('gen' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('gen' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('gen' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('gen', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'T_min':3, 'fit_min':7})
    stats_df.to_csv(path_out3, index=False)


def S_gen():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'gen')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'S_gen.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'S_gen_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    for g in ['100', '200', '300']:
        file_out = 'g' + g + '.csv'
        path_out = os.path.join(dir_out, file_out)
        folder_in = 'g' + g
        path = os.path.join(BASE_DIR, 'results', 'Solomon', folder_in)
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        column = []
        for m in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            for c in [0.6, 0.7, 0.8]:
                for folder in dirs:
                    file = folder + '_croOrd_mutShu_selTour_wC100.0_dC100.0_iS100_pS1000_cP'  + str(c) + '_mP' + str(m) + '_nG' + g + '.csv'
                    path_i = os.path.join(path, folder)
                    path_in = os.path.join(path_i, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    sub_df['Cx'] = c_column
                    m_column = str(m)
                    sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    # sub_df['inst'] = folder
                    column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(27, columns=['fitness'])
        df_min['gen'] = g
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby('gen', as_index=False)['time'].median()
    stats_df.columns = ['gen','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('gen', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['C_medi']= merged_min_df.groupby('gen', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('gen', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby('gen' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('gen', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['C_avg']  = merged_min_df.groupby('gen' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby('gen' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby('gen' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('gen', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'fitness':7, 'time':3})
    stats_df.to_csv(path_out3, index=False)


def A_gen_pop():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'Spop_Lgen')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'A_Spop_Lgen.csv'
    # file_out2 = 'A_pop.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    # file_out3 = 'A_Spop_Lgen_stats_all.csv'
    file_out3 = 'A_pop_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    # for p, g in zip(['10', '20', '30', '50', '70', '100', '200', '500', '625'], ['50000', '25000', '15000', '10000', '7000', '5000', '2500', '1000', '800']):
    for p, g in zip(['50000', '25000', '10000', '5000', '2000', '1000', '800'], ['10', '20', '50', '100', '250', '500', '625']):
        file_out = 'p' + p + 'g' + g + '.csv'
        path_out = os.path.join(dir_out, file_out)
        folder_in = 'p' + p + 'g' + g
        path = os.path.join(BASE_DIR, 'results', 'A', 'Spop_Lgen')
        column = []
        for m in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            for c in [0.6, 0.7, 0.8]:
                file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS' + p + '_cP' + str(c) + '_mP' + str(m) + '_nG' + g + '.csv'
                path_in = os.path.join(path, file)
                df = pd.read_csv(path_in)
                sub_df = pd.DataFrame([df.iloc[-2, 2]])
                sub_df.columns = ['fitness']
                c_column = str(c)
                sub_df['Cx'] = c_column
                m_column = str(m)
                sub_df['Mut'] = m_column
                sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(63, columns=['fitness'])
        df_min['pop'] = p
        df_min['gen'] = g
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby(['pop', 'gen'], as_index=False)['time'].median()
    stats_df.columns = ['population', 'genSize','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby(['pop', 'gen'], as_index=False)['fitness'].median().iloc[:,2]
    # stats_df['C_medi']= merged_min_df.groupby(['pop', 'gen'], as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby(['pop', 'gen'], as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby(['pop', 'gen'] , as_index=False)['time'].mean().iloc[:,2] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby(['pop', 'gen'], as_index=False)['fitness'].mean().iloc[:,2]
    # stats_df['C_avg']  = merged_min_df.groupby(['pop', 'gen'] , as_index=False)['Cx'].mean().iloc[:,2] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby(['pop', 'gen'] , as_index=False)['Mut'].mean().iloc[:,2] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby(['pop', 'gen'] , as_index=False)['time'].min().iloc[:,2]
    stats_df['fit_min']=merged_min_df.groupby(['pop', 'gen'], as_index=False)['fitness'].min().iloc[:,2]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':5, 'fit_avg':5, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'fitness':7, 'time':3})
    stats_df.to_csv(path_out3, index=False)


def S_gen_pop():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'Spop_Lgen')
    os.makedirs(dir_out, exist_ok=True)
    # file_out2 = 'S_Spop_Lgen.csv'
    file_out2 = 'S_Spop_Lgen_L.csv'
    # file_out2 = 'S_pop.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    # file_out3 = 'S_Spop_Lgen_stats_all.csv'
    file_out3 = 'S_Spop_Lgen_stats_L_all.csv'
    # file_out3 = 'S_pop_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    # for p, g in zip(['10', '20', '30', '50', '70', '100', '200', '300'], ['10000', '5000', '3300', '2000', '1400', '1000', '500', '333']):
    for p, g in zip(['10', '20', '40', '60', '100'], ['20000', '10000', '5000', '3300', '2000']):
    # for p, g in zip(['20000', '10000', '5000', '3300', '2000', '1000', '500'], ['10', '20', '40', '60', '100', '200', '400']):
        file_out = 'p' + p + 'g' + g + '.csv'
        path_out = os.path.join(dir_out, file_out)
        folder_in = 'p' + p + 'g' + g
        path = os.path.join(BASE_DIR, 'results', 'Solomon', 'Spop_Lgen')
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        column = []
        for m in [0.1, 0.2, 0.3]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
            for c in [0.6, 0.7, 0.8]:
                for folder in dirs:
                    file = folder + '_croOrd_mutShu_selTour_wC100.0_dC100.0_iS100_pS' + p + '_cP'  + str(c) + '_mP' + str(m) + '_nG' + g + '.csv'
                    path_i = os.path.join(path, folder)
                    path_in = os.path.join(path_i, file)
                    df = pd.read_csv(path_in)
                    sub_df = pd.DataFrame([df.iloc[-2, 2]])
                    sub_df.columns = ['fitness']
                    c_column = str(c)
                    sub_df['Cx'] = c_column
                    m_column = str(m)
                    sub_df['Mut'] = m_column
                    sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                    # sub_df['inst'] = folder
                    if folder[0] == 'C':
                        sub_df['inst'] = 11
                    elif folder[:2] == 'RC':
                        sub_df['inst'] = 33
                    else:
                        sub_df['inst'] = 22
                    column.append(sub_df)
        merged_dataframe = pd.concat(column, axis=0)
        # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
        df_min = merged_dataframe.nsmallest(45, columns=['fitness'])
        df_min['pop'] = p
        df_min['gen'] = g
        avg = merged_dataframe['fitness'].mean()
        medi = merged_dataframe['fitness'].median()
        df_min['F_avg'] = avg
        df_min['F_medi'] = medi
        df_min['T_avg'] = merged_dataframe['time'].mean()
        df_min['T_medi'] = merged_dataframe['time'].mean()
        aggr_list.append(df_min)
        # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    # merged_min_df = merged_min_df.round({'pop':0, 'gen':0, 'T_medi':3, 'T_avg':3, 'F_medi':6, 'F_avg':6, 'fitness':6, 'time':3})
    # merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False
    stats_df = merged_min_df.groupby(['inst', 'pop', 'gen'], as_index=False)['time'].median()
    stats_df.columns = ['inst','population','genSize','T_medi']
    stats_df['fit_medi']=merged_min_df.groupby(['inst', 'pop', 'gen'], as_index=False)['fitness'].median().iloc[:,3]
    # stats_df['C_medi']= merged_min_df.groupby('gen', as_index=False)['Cx'].median().iloc[:,1]   #
    # stats_df['M_medi']=merged_min_df.groupby('gen', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    stats_df['T_avg'] =merged_min_df.groupby(['inst', 'pop', 'gen'] , as_index=False)['time'].mean().iloc[:,3] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby(['inst', 'pop', 'gen'], as_index=False)['fitness'].mean().iloc[:,3]
    # stats_df['C_avg']  = merged_min_df.groupby(['inst', 'pop', 'gen'] , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # stats_df['M_avg'] = merged_min_df.groupby(['inst', 'pop', 'gen'] , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    stats_df['T_min'] =merged_min_df.groupby(['inst', 'pop', 'gen'] , as_index=False)['time'].min().iloc[:,3]
    stats_df['fit_min']=merged_min_df.groupby(['inst', 'pop', 'gen'], as_index=False)['fitness'].min().iloc[:,3]
    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':2, 'fit_avg':2, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'fitness':7, 'time':3})
    stats_df.to_csv(path_out3, index=False)


def A_pb():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'A', 'pb0')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'A_pb_min55.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'A_pb_stats.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    path = os.path.join(BASE_DIR, 'results', 'A', 'pb0')
    column = []
    for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #
        file_out = 'm' + str(m) + '.csv'
        path_out = os.path.join(dir_out, file_out)
        for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            file = 'A-n16-k4_croOrd_mutShu_selTour_wC100.0_dC100.0_iS16_pS50000_cP' + str(c) + '_mP' + str(m) + '_nG10.csv'
            path_in = os.path.join(path, file)
            df = pd.read_csv(path_in)
            sub_df = pd.DataFrame([df.iloc[-2, 2]])
            sub_df.columns = ['fitness']
            c_column = str(c)
            sub_df['Cx'] = c_column
            m_column = str(m)
            sub_df['Mut'] = m_column
            sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
            column.append(sub_df)
    merged_dataframe = pd.concat(column, axis=0)
    # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
    df_min = merged_dataframe.nsmallest(55, columns=['fitness'])
    avg = merged_dataframe['fitness'].mean()
    medi = merged_dataframe['fitness'].median()
    # df_min['fit_avg'] = avg
    # df_min['fit_medi'] = medi
    # df_min['T_avg'] = merged_dataframe['time'].mean()
    # df_min['T_medi'] = merged_dataframe['time'].mean()
    aggr_list.append(df_min)
    # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    merged_min_df = merged_min_df.round({'T_medi':3, 'T_avg':3, 'F_medi':4, 'F_avg':4, 'fitness':7, 'time':3})
    merged_min_df.to_csv(path_out2, index=False)
    merged_min_df = merged_min_df.astype(float) # , copy=False
    # stats_df = merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['time'].median()
    # stats_df.columns = ['Cx', 'Mut', 'T_medi']
    # stats_df['fit_medi']=merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['fitness'].median().iloc[:,2]
    # # stats_df['C_medi']= merged_min_df.groupby('tourSize', as_index=False)['Cx'].median().iloc[:,1]   #
    # # stats_df['M_medi']=merged_min_df.groupby('tourSize', as_index=False)['Mut'].median().iloc[:,1] # ['C_medi']
    # stats_df['T_avg'] =merged_min_df.groupby(['Cx', 'Mut'] , as_index=False)['time'].mean().iloc[:,2] # ['M_medi']
    # stats_df['fit_avg']=merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['fitness'].mean().iloc[:,2]
    # # stats_df['C_avg']  = merged_min_df.groupby('tourSize' , as_index=False)['Cx'].mean().iloc[:,1] # ['T_avg']
    # # stats_df['M_avg'] = merged_min_df.groupby('tourSize' , as_index=False)['Mut'].mean().iloc[:,1] # ['C_avg']
    # stats_df['T_min'] =merged_min_df.groupby(['Cx', 'Mut'] , as_index=False)['time'].min().iloc[:,2]
    # stats_df['fit_min']=merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['fitness'].min().iloc[:,2]

    # stats_df = merged_min_df.groupby('Cx', as_index=False)['time'].median()
    # stats_df.columns = ['Cx', 'T_medi']
    # stats_df['fit_medi']=merged_min_df.groupby('Cx', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['T_avg'] =merged_min_df.groupby('Cx' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    # stats_df['fit_avg']=merged_min_df.groupby('Cx', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['T_min'] =merged_min_df.groupby('Cx' , as_index=False)['time'].min().iloc[:,1]
    # stats_df['fit_min']=merged_min_df.groupby('Cx', as_index=False)['fitness'].min().iloc[:,1]
    stats_df = merged_min_df.groupby('Mut', as_index=False)['time'].median()
    stats_df.columns = ['Mut', 'T_medi']
    stats_df['fit_medi']=merged_min_df.groupby('Mut', as_index=False)['fitness'].median().iloc[:,1]
    stats_df['T_avg'] =merged_min_df.groupby('Mut' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby('Mut', as_index=False)['fitness'].mean().iloc[:,1]
    stats_df['T_min'] =merged_min_df.groupby('Mut' , as_index=False)['time'].min().iloc[:,1]
    stats_df['fit_min']=merged_min_df.groupby('Mut', as_index=False)['fitness'].min().iloc[:,1]

    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'T_min':3, 'fit_min':7})
    # stats_df.to_csv(path_out3, index=False)


def S_pb():
    dir_out = os.path.join(BASE_DIR, 'resultAnalysis', 'S', 'pb0')
    os.makedirs(dir_out, exist_ok=True)
    file_out2 = 'S_pb_min11.csv'
    path_out2 = os.path.join(dir_out, file_out2)
    file_out3 = 'S_pb_stats_all.csv'
    path_out3 = os.path.join(dir_out, file_out3)
    aggr_list = []
    path = os.path.join(BASE_DIR, 'results', 'Solomon', 'pb0')
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    column = []
    for m in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]: #, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

        file_out = 'm' + str(m) + '.csv'
        path_out = os.path.join(dir_out, file_out)
        for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for folder in dirs:
                file = folder + '_croOrd_mutInv_selTour_wC100.0_dC100.0_iS100_pS5000_cP' + str(c) + '_mP' + str(m) + '_nG40.csv'
                path_i = os.path.join(path, folder)
                path_in = os.path.join(path_i, file)
                df = pd.read_csv(path_in)
                sub_df = pd.DataFrame([df.iloc[-2, 2]])
                sub_df.columns = ['fitness']
                c_column = str(c)
                sub_df['Cx'] = c_column
                m_column = str(m)
                sub_df['Mut'] = m_column
                sub_df['time'] = pd.DataFrame([df.iloc[-1, 1]])
                # sub_df['inst'] = folder
                column.append(sub_df)
    merged_dataframe = pd.concat(column, axis=0)
    # df_min = merged_dataframe[merged_dataframe.fitness == merged_dataframe.fitness.min()]
    df_min = merged_dataframe.nsmallest(11, columns=['fitness'])
    avg = merged_dataframe['fitness'].mean()
    medi = merged_dataframe['fitness'].median()
    # df_min['F_avg'] = avg
    # df_min['F_medi'] = medi
    # df_min['T_avg'] = merged_dataframe['time'].mean()
    # df_min['T_medi'] = merged_dataframe['time'].mean()
    aggr_list.append(df_min)
    # merged_dataframe.to_csv(path_out, index=False)
    merged_min_df = pd.concat(aggr_list, axis=0)
    merged_min_df = merged_min_df.round({'T_medi':3, 'T_avg':3, 'F_medi':4, 'F_avg':4, 'fitness':9, 'time':3})
    merged_min_df.to_csv(path_out2, index=False)

    merged_min_df = merged_min_df.astype(float) # , copy=False

    stats_df = merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['time'].median()
    stats_df.columns = ['Cx', 'Mut', 'T_medi']
    stats_df['fit_medi']=merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['fitness'].median().iloc[:,2]
    stats_df['T_avg'] =merged_min_df.groupby(['Cx', 'Mut'] , as_index=False)['time'].mean().iloc[:,2] # ['M_medi']
    stats_df['fit_avg']=merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['fitness'].mean().iloc[:,2]
    stats_df['T_min'] =merged_min_df.groupby(['Cx', 'Mut'] , as_index=False)['time'].min().iloc[:,2]
    stats_df['fit_min']=merged_min_df.groupby(['Cx', 'Mut'], as_index=False)['fitness'].min().iloc[:,2]

    # stats_df = merged_min_df.groupby('Mut', as_index=False)['time'].median()
    # stats_df.columns = ['Mut', 'T_medi']
    # stats_df['fit_medi']=merged_min_df.groupby('Mut', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['T_avg'] =merged_min_df.groupby('Mut' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    # stats_df['fit_avg']=merged_min_df.groupby('Mut', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['T_min'] =merged_min_df.groupby('Mut' , as_index=False)['time'].min().iloc[:,1]
    # stats_df['fit_min']=merged_min_df.groupby('Mut', as_index=False)['fitness'].min().iloc[:,1]

    # stats_df = merged_min_df.groupby('Cx', as_index=False)['time'].median()
    # stats_df.columns = ['Cx', 'T_medi']
    # stats_df['fit_medi']=merged_min_df.groupby('Cx', as_index=False)['fitness'].median().iloc[:,1]
    # stats_df['T_avg'] =merged_min_df.groupby('Cx' , as_index=False)['time'].mean().iloc[:,1] # ['M_medi']
    # stats_df['fit_avg']=merged_min_df.groupby('Cx', as_index=False)['fitness'].mean().iloc[:,1]
    # stats_df['T_min'] =merged_min_df.groupby('Cx' , as_index=False)['time'].min().iloc[:,1]
    # stats_df['fit_min']=merged_min_df.groupby('Cx', as_index=False)['fitness'].min().iloc[:,1]

    stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':2, 'fit_avg':2, 'T_min':3, 'fit_min':7})
    # stats_df = stats_df.round({'T_medi':3, 'T_avg':3, 'fit_medi':7, 'fit_avg':7, 'fitness':7, 'time':3})
    # stats_df.to_csv(path_out3, index=False)


# S_pb()
# A_pb()
# S_gen_pop()
# A_gen_pop()
# S_gen()
# A_gen()
# S_tourSize()
S_selection()
# S_pop()
# A_selection()
# A_tour_size()
# A_20sets_aggregation()
# S_CM_operators()
# A_CM_operators()

# filenames = os.listdir(".")
# print(filenames)


# all_gen_aggregation()
# final_fit_aggregation()
# combine_final_fit()
# box_plot()

# for pop in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]:
#     line_plot(pop)
# line_plot_m(1000)
# line_plot_m(1100)
# line_plot_m(1200)
# line_plot_m(1300)
# line_plot_m(1400)
# line_plot_m(1500)
# line_plot_m(1600)
# line_plot_m(1700)
# line_plot_m(1800)
# line_plot_m(1900)
# line_plot_mut(2000)
# select_best_CxMt()
# line_plot_pop()
# aggregation_plot()
# group_by_pop()
# conda install -c anaconda plotly
