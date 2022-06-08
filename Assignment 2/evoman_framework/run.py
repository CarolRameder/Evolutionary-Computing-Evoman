import os, sys
import numpy as np
from enum import Enum

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, 'evoman')

from evolutionary_wrapper import EvolutionWrapper, CrossoverOperators, MutationOperators, SelectionOperators
from environment import Environment
from demo_controller import player_controller
import record_results as record


experiment_name = 'evolutionary_wrapper'

class Algorithm(Enum):
    ALGORITHM_ONE = '1',
    ALGORITHM_TWO = '2'

    
NUMBER_OF_GENERATIONS = 25
NUMBER_OF_POPULATION = 100
NUMBER_OF_RUNS = 10
NUMBER_OF_BEST_INDS_RUNS = 5
NUMBER_METHOD = Algorithm.ALGORITHM_TWO.value[0]
n_hidden_neurons = 10
NUM_CLONES = 4

GROUP_ENEMIES_ONE = [1,2,3,4]
GROUP_ENEMIES_TWO = [5,6,7,8]


env = Environment(experiment_name=experiment_name,
                  enemies=GROUP_ENEMIES_ONE,
                  multiplemode = "yes", 
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  contacthurt="player",
                  randomini="yes")
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

# run 5 times the best individual & return avg number
def run_exp(individual):
    sum_gain = 0
    sum_fitness = 0
    for _ in range(NUMBER_OF_BEST_INDS_RUNS):
        enemies, fitnesses, player_lifes, enemy_lifes, times = env.play(pcont=individual)
        gain_rank = np.mean(np.array(player_lifes) - np.array(enemy_lifes))
        fitness = np.mean(fitnesses)
        sum_gain = sum_gain + gain_rank
        sum_fitness = sum_fitness + fitness

    avg_gain = sum_gain/float(NUMBER_OF_BEST_INDS_RUNS)
    avg_fitness = sum_fitness/float(NUMBER_OF_BEST_INDS_RUNS)
    return avg_gain, avg_fitness


def get_evo_wrapper(method_number, run_number):
    if method_number == Algorithm.ALGORITHM_ONE.value[0]: 
        return EvolutionWrapper(
            individual_array_size=n_vars,
            population_size=NUMBER_OF_POPULATION,
            environment=env,
            mutation_operator= MutationOperators.SELF_ADAPTIVE_MUTATION,
            mutation_arguments = {
                'tau': 1 / np.sqrt(2*n_vars),
                'tau_prime': 1 / np.sqrt(2 * np.sqrt(n_vars))
            },
            crossover_operator=CrossoverOperators.MULTIPARENTS_UNIFORM_CROSSOVER,
            crossover_arguments={
                'double_offsprings': True
            },
            selection_operator=SelectionOperators.SEL_TOURNAMENT,
            selection_arguments = {
                'k': NUMBER_OF_POPULATION,
                'tournsize': 20,
                'fit_attr': 'fitness'
            },
            run_number=str(run_number),
            method_number=Algorithm.ALGORITHM_ONE.value[0],
            num_clones=NUM_CLONES)
    else:
        return EvolutionWrapper(
            individual_array_size=n_vars,
            population_size=NUMBER_OF_POPULATION,
            environment=env,
            mutation_operator= MutationOperators.SELF_ADAPTIVE_MUTATION,
            mutation_arguments = {
                'tau': 1 / np.sqrt(2*n_vars),
                'tau_prime': 1 / np.sqrt(2 * np.sqrt(n_vars))
            },
            crossover_operator=CrossoverOperators.MULTIPARENTS_UNIFORM_CROSSOVER,
            crossover_arguments={
                'double_offsprings': False
            },
            selection_operator=SelectionOperators.SEL_TOURNAMENT,
            selection_arguments={
                'k': NUMBER_OF_POPULATION,
                'tournsize': 20,
                'fit_attr': 'fitness'
            },
            run_number=str(run_number),
            method_number=Algorithm.ALGORITHM_TWO.value[0],
            num_clones=NUM_CLONES)

for run_number in range(1, NUMBER_OF_RUNS + 1): 
    evo_wrapper = get_evo_wrapper(NUMBER_METHOD, run_number)
    exp_best_gains = []
    exp_best_inds = []
    for i in range(NUMBER_OF_GENERATIONS):
        print('---------NEW GENERATION---------')
        print(f'Current gen: {i}')
        evo_wrapper.new_generation()
        # select the best individual
        current_best_ind, current_best_fit, current_best_gain = evo_wrapper.best_from_current_population()
        exp_best_gains.append(current_best_gain)
        exp_best_inds.append(current_best_ind)

    # select the best individual
    exp_best_gain = max(exp_best_gains)
    best_ind_index = exp_best_gains.index(exp_best_gain) # find the index of the best gain so as to find best_ind
    exp_best_ind = exp_best_inds[best_ind_index]

    # run 
    exp_final_gain_avg, exp_final_fitness = run_exp(exp_best_ind)

    # record data
    results = [{ 'gain_rank_avg': exp_final_gain_avg, 'fitness_avg': exp_final_fitness, 'run_number': run_number, 'method': NUMBER_METHOD, 'best_ind': exp_best_ind  }]
    record.record_rank_gain(env.enemies, NUMBER_METHOD, results)
