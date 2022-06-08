import os, json, pathlib, glob
from dataclasses import dataclass, asdict
import numpy as np
import pprint

@dataclass
class FightStats:
    """Save stats after a fight"""
    enemy: int
    fitness: float
    player_life: int
    enemy_life: int
    time: int

@dataclass
class Individual:
    """Class for saving individuals that beat at least one enemy"""
    idx: int
    fights: list
    weights: list

class DataHandler:

    @staticmethod
    def get_current_ind_index():
        return json.load(open("individuals2/counter.json"))["individual_index"]

    @staticmethod
    def update_current_ind_index():
        counter = json.load(open("individuals2/counter.json"))
        counter["individual_index"] = counter["individual_index"] + 1
        json.dump(counter, open("individuals2/counter.json", 'w'))

    @staticmethod
    def save_individual(individual, enemies, fitnesses, player_lifes, enemy_lifes, times):
        fights = []
        for idx, enemy in enumerate(enemies):
            fight = FightStats(
                enemy = enemy,
                fitness = fitnesses[idx],
                player_life = player_lifes[idx],
                enemy_life = enemy_lifes[idx],
                time = times[idx]
            )
            fights.append(fight)
        ind_index = DataHandler.get_current_ind_index()
        saved_ind = Individual(
            idx = ind_index,
            weights = individual[:],
            fights = fights
        )
        # Build dir name where the ind is saved
        dir_name = DataHandler.get_dir_name(ind_index)
        current_dir = pathlib.Path(__file__).parent.resolve()
        combined_dir = f"{current_dir}/individuals2/{dir_name}"

        # Create if it doesn't exist
        pathlib.Path(combined_dir).mkdir(parents=True, exist_ok=True)

        json.dump(asdict(saved_ind), open(f"{combined_dir}/ind_{ind_index}.json", 'w'), indent=4)
        DataHandler.update_current_ind_index()

    @staticmethod
    def get_dir_name(ind_index):
        range = DataHandler.get_index_range(ind_index)
        return f"Individuals2_{range}"

    @staticmethod
    def get_index_range(ind_index):
        lower_end = ind_index - (ind_index % 50)
        upper_end = lower_end + 49
        return f"{lower_end}-{upper_end}"

    @staticmethod
    def print_individual_indexes(enemies_beaten = 3, avg_fitness = False):
        all_individuals = DataHandler.get_all_individuals()
        for i in range (enemies_beaten, 9):
            print(f"----- {i} {'enemy' if i == 1 else 'enemies'} beaten -----")
            if not avg_fitness:
                individuals_indexes = [ind["idx"] for ind in all_individuals if DataHandler.enemies_beaten(ind) == i]
                print(sorted(individuals_indexes))
            else:
                individuals = [(ind["idx"], DataHandler.average_fitness(ind)) for ind in all_individuals if DataHandler.enemies_beaten(ind) == i]
                print(*sorted(individuals, key=lambda x:x[1]), sep='\n')


    @staticmethod
    def average_fitness(individual):
        return np.average([fight["fitness"] for fight in individual["fights"]])

    @staticmethod
    def enemies_beaten(individual):
        enemies_beaten = 0
        for fight in individual["fights"]:
            if fight["enemy_life"] == 0:
                enemies_beaten = enemies_beaten + 1
        return enemies_beaten

    @staticmethod
    def get_all_individuals():
        current_dir = pathlib.Path(__file__).parent.resolve()
        all_individuals = []
        for filename in glob.iglob(f"{current_dir}/individuals2/**", recursive=True):
            if os.path.isfile(filename) and filename.endswith(".json") and not (filename.endswith("counter.json") or filename.endswith("counter1.json")):
                with open(filename) as json_file:
                    individual = json.load(json_file)
                    all_individuals.append(individual)
        return all_individuals

    @staticmethod
    def get_individuals_that_beated_n_enemies(n):
        all_individuals = DataHandler.get_all_individuals()
        return [ind for ind in all_individuals if DataHandler.enemies_beaten(ind)]

#        return [ind for ind in all_individuals if DataHandler.enemies_beaten(ind) > n]

    @staticmethod
    def get_individuals_by_indexes(indexes):
        all_individuals = DataHandler.get_all_individuals()
        return [ind for ind in all_individuals if ind["idx"] in indexes]



