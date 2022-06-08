import os
import csv

# csv header
fieldnames = ['generation_number', 'min', 'max', 'avg', 'std']
fieldnames_boxplot = ['run_number', 'method', 'gain_rank_avg', 'fitness_avg', 'best_ind']
fieldnames_boxplot_testing_groups = ['method', 'gain_rank_avg', 'player_life_avg', 'enemy_life_avg', 'time_avg', 'ind_index']

FILE_PATH = os.path.abspath(os.path.dirname(__file__))
FILE_NAME_OUTPUTS = '/Outputs2'
FILE_NAME_ENEMY_NUMBER = '/enemy_{0}'
FILE_NAME_METHOD_NUMBER = '/method_{0}'
FILE_NAME_RESULTS_CSV = '/run_{0}.csv'
FILE_NAME_BOXPLOT_CSV = '/boxplot.csv'

def add_fieldnames(filename, fieldnames): 
    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def check_file_existance(foldername, filename, fieldnames): 
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        add_fieldnames(filename, fieldnames)
    elif not os.path.exists(filename):
        add_fieldnames(filename, fieldnames)


def record_results(enemy_number, method_number, run_number, results):
    foldername = FILE_PATH + FILE_NAME_OUTPUTS + FILE_NAME_ENEMY_NUMBER.format(enemy_number) + FILE_NAME_METHOD_NUMBER.format(method_number)
    filename = foldername + FILE_NAME_RESULTS_CSV.format(run_number)

    check_file_existance(foldername, filename, fieldnames)

    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(results)

def record_rank_gain(enemy_number, method_number, results):
    foldername = FILE_PATH + FILE_NAME_OUTPUTS + FILE_NAME_ENEMY_NUMBER.format(enemy_number) + FILE_NAME_METHOD_NUMBER.format(method_number)
    filename = foldername + FILE_NAME_BOXPLOT_CSV

    check_file_existance(foldername, filename, fieldnames_boxplot)

    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames_boxplot)
        writer.writerows(results)
        
def record_rank_gain_testing_groups(enemy_group, method_number, results):
    foldername = FILE_PATH + FILE_NAME_OUTPUTS + FILE_NAME_ENEMY_NUMBER.format(enemy_group) + FILE_NAME_METHOD_NUMBER.format(method_number)
    filename = foldername + FILE_NAME_BOXPLOT_CSV

    check_file_existance(foldername, filename, fieldnames_boxplot_testing_groups)

    with open(filename, 'a', encoding='UTF8', newline='\n') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames_boxplot_testing_groups)
        writer.writerows(results)