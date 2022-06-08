import sys, random

sys.path.insert(0, 'evoman')

# EVOMAN
from environment import Environment

# DEAP
from deap import base, creator, tools

import numpy as np
import record_results as record

import logging
from enum import Enum

from DataHandler import DataHandler

INDIVIDUAL_LOWER_BOUND = -1
INDIVIDUAL_UPPER_BOUND = 1
EPSILON_ZERO = 0.05

def _multiple_parent_diagonal_crossover(selected_parents, parents_per_offspring: int, population_size, toolbox):
    """
    Takes in a list of selected parents and performs n-point crossover with multiple parents
    self._population_size
    """
    splitted_parents = np.array_split(selected_parents, len(selected_parents) / parents_per_offspring)
    toolbox_offsprings = toolbox.population(n=population_size)
    offspring_idx = 0
    for mating_parents in splitted_parents:
        mating_parents_splitted_values = []
        for mating_parent in mating_parents:
            mating_parents_splitted_values.append(np.array_split(mating_parent, parents_per_offspring))
        for child_nbr in range(len(mating_parents)):
            offspring = []
            for i in range(parents_per_offspring):
                offspring.append(mating_parents_splitted_values[(child_nbr+i) % parents_per_offspring][i])
            toolbox_offsprings[offspring_idx][:] = np.concatenate(offspring)
            offspring_idx = offspring_idx + 1
    return toolbox_offsprings

def multiple_parent_uniform_crossover(selected_parents, parents_per_offspring: int, population_size, toolbox, individual_array_size, double_offsprings):
    """
    Takes in a list of selected parents and performs uniform crossover with multiple parents
    self._population_size
    """
    random.shuffle(selected_parents)
    splitted_parents_weights = np.array_split(selected_parents, len(selected_parents) / parents_per_offspring)
    splitted_parents_sigmas = np.array_split([ind.sigmas for ind in selected_parents], len(selected_parents) / parents_per_offspring)
    splitted_parents_combined = zip(splitted_parents_weights, splitted_parents_sigmas)
    toolbox_offsprings = toolbox.population(n=population_size if not double_offsprings else population_size * 2)
    offspring_idx = 0
    for mating_parents in list(splitted_parents_combined):
        for child_nbr in range(len(mating_parents[0]) if not double_offsprings else len(mating_parents[0]) * 2):
            offspring_weights = []
            offspring_sigmas = []
            for i in range(individual_array_size):
                selected_parent_index = random.randint(0, len(mating_parents[0]) - 1)
                offspring_weights.append(mating_parents[0][selected_parent_index][i])
                offspring_sigmas.append(mating_parents[1][selected_parent_index][i])
            toolbox_offsprings[offspring_idx][:] = offspring_weights
            toolbox_offsprings[offspring_idx].sigmas[:] = offspring_sigmas
            offspring_idx = offspring_idx + 1
    return toolbox_offsprings

def _self_adaptive_mutation(individual, tau, tau_prime):
    # Change Sigma values first
    independent_draw = np.random.normal(loc=0, scale=1.0)
    for i in range(len(individual.sigmas)):
        coordinate_draw = np.random.normal(loc=0, scale=1.0)
        individual.sigmas[i] = max(individual.sigmas[i] * np.exp(tau_prime * independent_draw
                                                                 + tau * coordinate_draw), EPSILON_ZERO)
        individual[i] = individual[i] + individual.sigmas[i] * coordinate_draw

class CrossoverOperators(Enum):
    CX_ONE_POINT = tools.cxOnePoint
    CX_TWO_POINT = tools.cxTwoPoint
    CX_UNIFORM = tools.cxUniform
    CX_PARTIALY_MATCHED = tools.cxPartialyMatched
    CX_UNIFORM_PARTIALY_MATCHED = tools.cxUniformPartialyMatched
    CX_ORDERED = tools.cxOrdered
    CX_BLEND = tools.cxBlend
    CX_ES_BLEND = tools.cxESBlend
    CX_ES_TWO_POINT = tools.cxESTwoPoint
    CX_SIMULATED_BINARY = tools.cxSimulatedBinary
    CX_SIMULATED_BINARY_BOUNDED = tools.cxSimulatedBinaryBounded
    CX_MESSY_ONE_POINT = tools.cxMessyOnePoint
    MULTIPARENTS_UNIFORM_CROSSOVER = multiple_parent_uniform_crossover


# Not used now
class MutationOperators(Enum):
    MUT_GAUSSIAN = tools.mutGaussian
    MUT_SHUFFLE_INDEXES = tools.mutShuffleIndexes
    MUT_FLIP_BIT = tools.mutFlipBit
    MUT_POLYNOMIAL_BOUNDED = tools.mutPolynomialBounded
    MUT_UNIFORM_INT = tools.mutUniformInt
    MUT_ES_LOG_NORMAL = tools.mutESLogNormal
    SELF_ADAPTIVE_MUTATION = _self_adaptive_mutation
    


class SelectionOperators(Enum):
    SEL_TOURNAMENT = tools.selTournament
    SEL_ROULETTE = tools.selRoulette
    SEL_NSGA2 = tools.selNSGA2
    SEL_NSGA3 = tools.selNSGA3
    SEL_SPEA2 = tools.selSPEA2
    SEL_RANDOM = tools.selRandom
    SEL_BEST = tools.selBest
    SEL_WORST = tools.selWorst
    SEL_TOURNAMENT_DCD = tools.selTournamentDCD
    SEL_DOUBLE_TOURNAMENT = tools.selDoubleTournament
    SEL_STOCHASTIC_UNIVERSAL_SAMPLING = tools.selStochasticUniversalSampling
    SEL_LEXICASE = tools.selLexicase
    SEL_EPSILON_LEXICASE = tools.selEpsilonLexicase
    SEL_AUTOMATIC_EPSILON_LEXICASE = tools.selAutomaticEpsilonLexicase

class EvolutionWrapper:

    def __init__(self, individual_array_size, mutation_operator: MutationOperators, mutation_arguments: dict,
                 population_size: int, environment: Environment, selection_operator: SelectionOperators,
                 selection_arguments: dict, run_number: str, method_number: str, crossover_operator: CrossoverOperators,
                 crossover_arguments: dict, num_clones: int):
        self._current_pop = []
        self._individual_array_size = individual_array_size
        self._mutation_operator = mutation_operator
        self._mutation_arguments = mutation_arguments
        self._population_size = population_size
        self._env = environment
        self._crossover_operator = crossover_operator
        self._crossover_arguments = crossover_arguments
        self._selection_operator = selection_operator
        self._selection_arguments = selection_arguments
        self._run_number = run_number
        self._method_number = method_number

        self._generation_number = 0
        self._enemies_beaten_in_current_run = 0
        self._total_enemies_fought = 0
        self._num_clones = num_clones

        self._good_individuals = []

        # Logger
        logging.basicConfig(
            format="[%(asctime)s] %(levelname)s: %(message)s",
            datefmt='%H:%M:%S',
            handlers=[
                # logging.FileHandler(f'{self.saves_folder}/Logs.txt'),
                logging.StreamHandler()
            ]
        )

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)

        self._toolbox = base.Toolbox()
        self.__initialize_toolbox()
        self.__initialize_population()

    def new_generation(self):
        # Reset
        self._enemies_beaten_in_current_run = 0
        self._total_enemies_fought = 0
        # Update generation number
        self._generation_number = self._generation_number + 1

        # Select the next generation individuals
        selected_parents = self._toolbox.select(self._current_pop)

        # Create new generation
        # offsprings = self._multiple_parent_diagonal_crossover(selected_parents, parents_per_offspring=4)
        offsprings = self._create_offsprings(selected_parents)

        # Mutate offsprings and evaluate them
        for offspring in offsprings:
            self._toolbox.mutate(offspring)
            self._toolbox.evaluate_fitness_and_gain(offspring)

        if len(offsprings) > self._population_size:
            # Select best offsprings
            offsprings = self._toolbox.select_best(offsprings)

        self._logger.info("Evaluated %i individuals" % len(offsprings))
        self._current_pop = offsprings
        self._evaluate_population()

    def _clear_fitness_and_gain(self, *individuals):
        for individual in individuals:
            del individual.fitness.values
            del individual.gain

    # Crossover Operations
    def _create_offsprings(self, selected_parents):
        if (self._crossover_operator == CrossoverOperators.MULTIPARENTS_UNIFORM_CROSSOVER):
            return self._toolbox.crossover(
                selected_parents=selected_parents, 
                parents_per_offspring=4,  
                population_size=self._population_size, 
                toolbox=self._toolbox, 
                individual_array_size=self._individual_array_size)
        else: 
            offsprings_one = list(map(self._toolbox.clone, selected_parents))
            offsprings_two = list(map(self._toolbox.clone, selected_parents))
            for offspring_one, offspring_two, offspring_three, offspring_four in zip(offsprings_one[::2], offsprings_one[1::2],
                                                    offsprings_two[:(len(offsprings_two)//2)], offsprings_two[(len(offsprings_two)//2):]):
                self._toolbox.crossover(offspring_one, offspring_two)
                self._toolbox.crossover(offspring_three, offspring_four)
                self._clear_fitness_and_gain(offspring_one, offspring_two, offspring_three, offspring_four)
            offsprings = offsprings_one + offsprings_two if self._crossover_arguments['double_offsprings'] else offsprings_one
            return offsprings

    def _evaluate_population(self):
        fits = [ind.fitness.values[0] for ind in self._current_pop]
        self._logger.info("Min %s" % min(fits))
        self._logger.info("Max %s" % max(fits))
        self._logger.info("Avg %s" % np.average(fits))
        self._logger.info("Std %s" % np.std(fits))
        self._logger.info(f"Enemies beaten: {self._enemies_beaten_in_current_run} / {self._total_enemies_fought} ≈ {round(float(self._enemies_beaten_in_current_run) / self._total_enemies_fought * 100, 2)}%")
        
        results = [{ 'min': min(fits), 'max': max(fits), 'avg': np.average(fits), 'std': np.std(fits), 'generation_number': self._generation_number }]
        record.record_results(self._env.enemies, self._method_number, self._run_number , results)

    def best_from_current_population(self):
        best_ind = tools.selBest(self._current_pop, 1)[0]
        return best_ind, best_ind.fitness.values, best_ind.gain

    # Get fitness and gain rank for a single individual
    def _fitness_and_gain(self, individual):
        #enemies, fitnesses, player_lifes, enemy_lifes, times = self._env.play(pcont=individual)

        # Save if at least one enemy was beaten
        #if enemy_lifes.count(0) > 4:
        #    DataHandler.save_individual(
        #        individual=individual,
        #        enemies=enemies,
        #        fitnesses=fitnesses,
        #        player_lifes=player_lifes,
        #        enemy_lifes=enemy_lifes,
        #        times=times
        #    )

        #individual.gain = np.mean(np.array(player_lifes) - np.array(enemy_lifes))
        #individual.fitness.values = [np.mean(fitnesses)]
        #self._enemies_beaten_in_current_run = self._enemies_beaten_in_current_run + enemy_lifes.count(0)
        #self._total_enemies_fought = self._total_enemies_fought + len(enemies)
        fitness_avg, player_life_avg, enemy_lifes_avg, time_avg = self._env.play(pcont=np.array(individual))
        individual.gain = player_life_avg - enemy_lifes_avg
        individual.fitness.values = [fitness_avg]


    # Mutate a single individual
    def _mutation(self, individual):
        for i in range(len(individual)):
            if random.random() < self._mutation_probability:
                # Add a random value between -mutation_change_value and +mutation_change_value in the bounds of
                # INDIVIDUAL_LOWER_BindpbOUND and INDIVIDUAL_UPPER_BOUND
                mutation_value = np.random.uniform(low=-self._mutation_change_value, high=self._mutation_change_value)
                individual[i] = max(min(INDIVIDUAL_UPPER_BOUND, individual[i] + mutation_value), INDIVIDUAL_LOWER_BOUND)

    # Maybe change parameters depending on input at initialization
    def __initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", base=list, sigmas=list, fitness=creator.FitnessMax, gain=int)

        self._toolbox.register("attr_weight_range", random.uniform, -1, 1)
        self._toolbox.register("individual", tools.initRepeat, creator.Individual, self._toolbox.attr_weight_range,
                               self._individual_array_size)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

        self._toolbox.register("evaluate_fitness_and_gain", self._fitness_and_gain)
        self._toolbox.register("crossover", self._crossover_operator, **self._crossover_arguments)
        self._toolbox.register("mutate", self._mutation_operator, **self._mutation_arguments)
        self._toolbox.register("select", self._selection_operator,  **self._selection_arguments)
        self._toolbox.register("select_best", SelectionOperators.SEL_BEST,  k = self._population_size)

    def _initialize_sigma(self, individual):
        sigma_list = []
        for i in range(self._individual_array_size):
            sigma_val = np.random.uniform(low=EPSILON_ZERO, high=2*EPSILON_ZERO)
            sigma_list.append(sigma_val)
        individual.sigmas = sigma_list


    def __initialize_population(self):
        self._current_pop = self._toolbox.population(n=self._population_size)
        self._logger.info("Start of evolution")

        # Evaluate the entire population
        for individual in self._current_pop:
            self._initialize_sigma(individual)
            self._toolbox.evaluate_fitness_and_gain(individual)

        self._evaluate_population()
