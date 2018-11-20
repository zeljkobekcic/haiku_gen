import re

import numpy as np
from argparse import ArgumentParser
from os.path import expanduser
from os import listdir
from numpy import array, random as random
from pickle import dump


def availiable_data(path: str):
    return listdir(path)


def load_available_data(path: str):
    data = availiable_data(path)

    def load_file_helper(filename):
        with open(f'{filename}', 'r') as infile:
            return infile.read().strip()

    for file in data:
        print(file)
        load_file_helper(f'{path}/{file}')

    return array([load_file_helper(f'{path}/{file}') for file in data])


def save_to(output_path: str, object):
    with open(output_path, 'wb') as outfile:
        dump(object, outfile)


class PreprocessingArgumentParser(object):

    def __init__(self):
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--output', type=str, metavar='OUT_DIR',
                                     help='Absolute path to the output directory. Will be created if not existing')
        argument_parser.add_argument('--input', type=str, metavar='INP_DIR',
                                     help='Absolute path to the input directory with the raw data.')

        parser = argument_parser.parse_args()

        self.argument_parser = argument_parser
        self.parser = parser

    def get_input(self):
        input = expanduser(self.parser.input)
        return input

    def get_output(self):
        output = expanduser(self.parser.output)
        return output


def create_split(data_size, split_ratio=(.5, .5), random_seed=123):
    random.seed(random_seed)
    permutation = random.permutation(data_size)
    train = permutation[:int(data_size * split_ratio[0])]
    test = permutation[int(data_size * split_ratio[1]):]

    return train, test


def create_train_test_data(data, split_ratio=(.5, .5), random_seed=123):
    train_index, test_index = create_split(len(data), split_ratio, random_seed)
    return np.array(data)[train_index], np.array(data)[test_index]


def is_too_short(datapoint: str, size=10) -> bool:
    return len(datapoint) < size


def remove_special_characters(data: str) -> str:
    regex = r'[^a-zA-Z\ \n]'
    return str(re.sub(regex, '', data))


def string_to_lower(string: str) -> str:
    return string.lower()


def to_short_and_none(string: str) -> str:
    if is_too_short(string):
        return None
    else:
        return string


def remove_none(data: list) -> list:
    return [x for x in data if x is not None]