import glob, os, pathlib
from DataHandler import DataHandler
#DataHandler.print_individual_indexes()

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser()


# Optional argument
parser.add_argument('--enemies_beaten', type=int,
                    help='An optional integer argument')

# Switch
parser.add_argument('--avg_fitness', action='store_true',
                    help='A boolean switch')

args = parser.parse_args()
DataHandler.print_individual_indexes(args.enemies_beaten, args.avg_fitness)
