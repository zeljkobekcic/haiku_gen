import re
import numpy as np
from os import listdir
from numpy import array, random as random
from pickle import dump
from hyphenate import hyphenate_word


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
    regex = r'[^a-zA-Z \n]'
    return str(re.sub(regex, '', data))


def string_to_lower(string: str) -> str:
    return string.lower()


def too_short_except_none(string: str) -> str:
    if is_too_short(string):
        return None
    else:
        return string


def remove_none(data: list) -> list:
    return [x for x in data if x is not None]


def verify_haiku_except_none(haiku):
    if haiku is None:
        return None
    else:
        eval = verify_haiku(haiku)
        if not eval:
            return None
        else:
            return eval


def verify_haiku(haiku: str) -> bool:
    lines = [l.split(' ') for l in haiku.splitlines()]
    syllables_per_line = [count_syllables_for_line(l) for l in lines]
    valid_haiku = [5, 7, 5]

    return syllables_per_line == valid_haiku


def count_syllables_for_line(words: [str]) -> int:
    return sum(map(count_syllables, words))


def count_syllables(word: str) -> int:
    hypenated = hyphenate_word(word)
    hypens = len(hypenated)
    return hypens


def whitespace_character_groups(text: str) -> str:
    special_chars = '([^a-zA-Z0-9 ]+)'
    alpha_num = '([a-zA-Z0-9]+)'

    whitespaced = re.sub(special_chars + alpha_num, r'\1 \2', text)
    whitespaced = re.sub(alpha_num + special_chars, r'\1 \2', whitespaced)
    return whitespaced


def intersperse(iterable, delimiter):
    """
    TAKEN FROM:
    https://stackoverflow.com/questions/5655708/python-most-elegant-way-to-intersperse-a-list-with-an-element
    :return:
    """
    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delimiter
        yield x


def split_by_syllables(text: str) -> str:
    splitted = intersperse(text.split(' '), ' ')
    return [s for w in splitted for s in hyphenate_word(w)]


def split_by_words(text: str) -> str:
    splitted = intersperse(text.split(' '), ' ')
    return [s for s in splitted]


def split_by_characters(text: str) -> str:
    return list(text)
