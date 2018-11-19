import os
import re
import numpy as np
import numpy.random as random

from pickle import dump
from functools import reduce
from multiprocessing import Pool


def availiable_data(path):
    return os.listdir(path)


def load_available_data(path):
    data = availiable_data(path)

    def load_file_helper(filename):
        with open(f'{filename}', 'r') as infile:
            return infile.read().strip()

    for file in data:
        print(file)
        load_file_helper(f'{path}/{file}')

    return np.array([load_file_helper(f'{path}/{file}') for file in data])


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


if __name__ == "__main__":

    data = load_available_data('../datasets/raw')[:100]

    pipeline = [remove_special_characters,
                lambda x: x.lower(),
                lambda x: x if x is not is_too_short(x) else None]


    def apply_pipeline(haiku: str):
        return reduce(lambda x, f: f(x), pipeline, haiku)


    with Pool(4) as p:
        mapped = p.map(apply_pipeline, data)

    only_valid_haikus = np.array([x for x in mapped if x is not None])
    train, test = create_split(len(only_valid_haikus))

    with open('data/train.pkl', 'wb') as outfile:
        dump(only_valid_haikus[train], outfile)

    with open('data/test.pkl', 'wb') as outfile:
        dump(only_valid_haikus[test], outfile)

    some_array = np.arange(0, 10)
    

