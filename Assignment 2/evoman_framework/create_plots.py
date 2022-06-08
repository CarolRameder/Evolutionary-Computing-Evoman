import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from scipy import stats
import csv

from numpy.lib.function_base import average 

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_NAME_OUTPUTS = '/Outputs'
FILE_ENEMY_NUMBER = '/enemy_'
FILE_METHOD_NUMBER = '/method_'
FILE_RESULTS = '/run_*.csv'
FILE_BOXPLOT_DATA = '/boxplot_testing.csv'
FILE_TTEST = '/TtestResults.csv'

method_numbers = ['1', '2']

enemy_groups = ['group1', 'group2']
plt.rcParams.update({'font.size': 16})

### LINE PLOTS:

# The mean and maximum fitness value of each run - based on the number of generations
# All ten runs calculate the mean value of means and maximums - draw your plot based on these numbers
# Thre is a 2d plot that shows the mean fitness and maximum fitness of each generation
# Calculate the deviations of the maximum and deviation of the mean - based on your data and add them to your plots.


### BOXPLOTS:

# Then you select the best individual of each run. The best selection process can be done by fitness, gain ranks. For the competition, they are supposed to compare based on ranks
# Then you test these best individuals of each run 5 times - So find the best controller of EA1-Run1 and test it 5 times and do the same for other runs
# Then you need to calculate the mean value of these 5 tests. So now you have one value for EA1-Run1, one for EA1-run2,…, one for EA1-run10. 
# Now you have to all these means values for EA1 in the same pot and then draw your box plot to see how much they are good.

def combine_results(method_numbers, enemy_number):

    csvfiles = {}
    data = {}
    
    for method_number in method_numbers:
        file_name = FILE_PATH + FILE_NAME_OUTPUTS + FILE_ENEMY_NUMBER + enemy_number + FILE_METHOD_NUMBER + method_number + FILE_RESULTS
        csvfiles[method_number] = glob.glob(file_name)

        data.update({
            method_number: {}
        })

        for files in csvfiles[method_number]:
            df = pd.read_csv(files)
            for _, row in df.iterrows():
                if row['generation_number'] in data[method_number] and 'max_values' in data[method_number][row['generation_number']]:
                    data[method_number].update({
                        row['generation_number']: {
                            'max_values': (data[method_number][row['generation_number']]['max_values']) + [float(row['max'])], # max fitness values
                            'avg_values': (data[method_number][row['generation_number']]['avg_values']) + [float(row['avg'])]  # avg fitness values
                        }
                    })
                else:
                    data[method_number].update({
                        row['generation_number']: {
                            'max_values': [float(row['max'])], # max fitness values
                            'avg_values': [float(row['avg'])]  # avg fitness values
                        }
                    })
    return data             


def get_boxplot_data(method_numbers, enemy_number):

    csvfiles = {}
    data = {}
    
    for method_number in method_numbers:
        file_name = FILE_PATH + FILE_NAME_OUTPUTS + FILE_ENEMY_NUMBER + enemy_number + FILE_METHOD_NUMBER + method_number + FILE_BOXPLOT_DATA
        csvfiles[method_number] = glob.glob(file_name)
        column_number = 'EA%s' % method_number
        data.update({
            column_number: []
        })

        for files in csvfiles[method_number]:
            df = pd.read_csv(files)
            for _, row in df.iterrows():
                data[column_number].append(float(row['gain_rank_avg'])) # avg gain_rank value
    return data             


def create_lineplot_per_enemy(data, method_numbers, enemy_number):
    fmt=["-", ":"]
    color=["b", "m"]

    avg = {}
    max = {}
    std_max = {}
    std_avg = {}

    for i in method_numbers: 
        avg.update({ 'x' + i: [], 'y' + i: [] })
        max.update({ 'x' + i: [], 'y' + i: [] })
        std_max.update({ i: [] })
        std_avg.update({ i: [] })

        for gen in data[i]:
            # method i average
            avg['x' + i].append(gen)
            avg['y' + i].append(np.average(data[i][gen]['avg_values']))
            # method i maximum
            max['x' + i].append(gen)
            max['y' + i].append(np.average(data[i][gen]['max_values']))
            # method i std for maximum
            std_max[i].append(np.std(data[i][gen]['max_values']))
            # method i std for mean
            std_avg[i].append(np.std(data[i][gen]['avg_values']))

        # plotting the method 1 points 
        plt.errorbar(avg['x' + i], avg['y' + i], std_avg[i], label = "EA%s average" % i, alpha=.65, fmt=fmt[int(i)-1], color=color[int(i)-1], capsize=1, capthick=1, elinewidth=0.4, ecolor='black')
        data_avg_fill_between = {
            'x': avg['x' + i],
            'y1': [y - e for y, e in zip(avg['y' + i], std_avg[i])], 
            'y2': [y + e for y, e in zip(avg['y' + i], std_avg[i])]}
        plt.fill_between(**data_avg_fill_between, alpha=.05, color=color[int(i)-1])
        # plotting the method 2 points 
        plt.errorbar(max['x' + i], max['y' + i], std_max[i], label = "EA%s maximum" % i, alpha=.65, fmt=fmt[int(i)-1], color=color[int(i)-1], capsize=1, capthick=1, elinewidth=0)
        data_max_fill_between = {
            'x': max['x' + i],
            'y1': [y - e for y, e in zip(max['y' + i], std_max[i])],
            'y2': [y + e for y, e in zip(max['y' + i], std_max[i])]}
        plt.fill_between(**data_max_fill_between, alpha=.05, color=color[int(i)-1])
    plt.xlabel('Generation')
    # Set the y axis label of the current axis.
    plt.ylabel('Fitness')
    # Set a title of the current axes.
    plt.title('Performance of %s' % enemy_number)
    # show a legend on the plot
    plt.legend(prop={'size': 6})
    # Display a figure.
    plt.show()


def create_boxplot_per_enemy(data, method_numbers, enemy_number):
    df = pd.DataFrame(data=data)
    columns = ['EA%s' % index for index in method_numbers]
    df.boxplot(column=columns)

    # Set the y axis label of the current axis.
    plt.ylabel('Individual Gain')
    # Set a title of the current axes.
    plt.title('Best Individuals of Training %s' % enemy_number)
    # show a legend on the plot
    # plt.legend()
    # Display a figure.
    plt.show()


# def do_boxplot_t_test(data, EAs, enemy_number):
#     filename = FILE_PATH + FILE_NAME_OUTPUTS + FILE_ENEMY_NUMBER + enemy_number + FILE_TTEST

#     Alg1 = data['EA' + EAs[0]]
#     Alg2 = data['EA' + EAs[1]]

#     ttest_results = stats.ttest_ind(Alg1, Alg2)
#     results = [{'enemy_group': enemy_number, 'pvalue': str(ttest_results.pvalue), 'statistic': str(ttest_results.statistic)}]
#     with open(filename, 'a', encoding='UTF8', newline='\n') as file:
#         writer = csv.DictWriter(file, fieldnames=['enemy_group', 'pvalue', 'statistic'])
#         writer.writeheader()
#         writer.writerows(results)

def ttest(alg1, alg2):
    filename = FILE_PATH + FILE_NAME_OUTPUTS + FILE_TTEST

    ttest_results = stats.ttest_ind(alg1, alg2)
    results = [{'pvalue': str(ttest_results.pvalue), 'statistic': str(ttest_results.statistic)}]
    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=['pvalue', 'statistic'])
        writer.writeheader()
        writer.writerows(results)


alg1 = []
alg2 = []
for enemy_number in enemy_groups:
    data_lineplot = combine_results(method_numbers, enemy_number)
    data_boxplot = get_boxplot_data(method_numbers, enemy_number)
    alg1 = alg1 + data_boxplot['EA' + method_numbers[0]]
    alg2 = alg2 + data_boxplot['EA' + method_numbers[1]]
    create_lineplot_per_enemy(data_lineplot, method_numbers, enemy_number)
    create_boxplot_per_enemy(data_boxplot, method_numbers, enemy_number)
    # do_boxplot_t_test(data_boxplot, method_numbers, enemy_number)


ttest(alg1, alg2)