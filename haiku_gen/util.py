import pickle as p
import numpy as np

from functools import reduce, partial
from keras.utils import to_categorical


def load_data_from(path):
    print(path)
    with open(path, 'rb') as infile:
        return p.load(infile)


def load_data(method):

    local_path = f'data/train_{method}.pkl'
    floydhub_path = f'/data/train_{method}.pkl'

    try:
        return load_data_from(local_path)
    except FileNotFoundError:
        return load_data_from(floydhub_path)


def load_test_data(method):
    local_path = f'data/train_{method}.pkl'
    floydhub_path = f'/data/train_{method}.pkl'

    try:
        return load_data_from(local_path)
    except FileNotFoundError:
        return load_data_from(floydhub_path)


def encode_datapoint(datapoint: list, mapping: dict) -> list:
    return to_categorical([mapping[d] for d in datapoint],
                          len(mapping),
                          dtype='int')


def encode_data(data: [list], mapping: dict) -> [list]:
    encoded = [encode_datapoint(datapoint, mapping) for datapoint in data]
    return np.array(encoded)


#TODO:
# ALL BELOW THIS NEEDS TO BE REWRITTEN
# WITH TDD <3

def character_set_on_data(data: list) -> dict:
    """
    Creates a set of characters over a collection of strings.
    :param data: A collection of strings.
    :return: The used characters in the collection of strings.
    """
    return reduce(lambda x, y: x | y, [set(x_i) for x_i in data])


def mapping_char_int(data: [str]) -> ({str: int}, {int: str}):
    """
    Creates two mappings. One from characters to numbers and one from numbers to characters.
    :param data: A collection os strings.
    :return: Two dictionaries.
    """
    char_set = character_set_on_data(data)
    char_2_int = {c: i for i, c in enumerate(char_set)}
    int_2_char = {i: c for i, c in enumerate(char_set)}

    return char_2_int, int_2_char


def create_timecell(datapoint: np.array, offset: int, window: int) -> np.array:
    """
    Helper function to create timecells for the recurrent neural networks.

    :param datapoint:
    :param offset:
    :param window:
    :return: A timecell from a datapoint.
    """
    timecell = datapoint[offset:offset + window]
    return timecell


def create_timeseries(datapoint: np.array, window: int) -> np.array:
    """
    A helper function to create timeseries from datapoints.

    :param datapoint:
    :param window:
    :return: A timeseries from a datapoint
    """

    def timecell(offset):
        return create_timecell(datapoint, offset, window)

    timecell_num = len(datapoint) - window + 1
    timeseries = np.array([timecell(i) for i in range(timecell_num)])
    return timeseries


def create_next_tensor(current_datapoint: np.array, predicted: np.array):
    array_length = predicted.shape[0]
    next_seq_element = list(current_datapoint[1, 1, array_length:]) + predicted
    as_tensor = np.array(next_seq_element).reshape((1, 1, -1))
    return as_tensor


def unify_data(data: [str], length=None) -> [str]:
    if length is None:
        length = max([len(d) for d in data])
    same_length = ["<BGN>" + d + "<END>" + (" " * (length + 1 - len(d))) for d in data]
    return same_length


def create_y_for(datapoint: [str]) -> [str]:
    try:
        y = [d[-1] for d in datapoint[1:]]
    except IndexError:
        print(datapoint)
    y.append(' ')
    return y


def save_model(model, naming_params):
    import os
    os.makedirs('model', exist_ok=True)
    model_name = 'arch={arch}-epochs={epochs}-num_elements={num_elements}-window={window}-batch_size={batch_size}.hdf5'
    model.save('model/' + model_name.format(**naming_params))


class TimeseriesEncoder:

    def __init__(self, x_data, window=5):
        x_data_unified = unify_data(x_data)
        self.char2int, self.int2char = mapping_char_int(x_data_unified)
        self.length = max(map(len, x_data))
        self.window = window

    def transform(self, x_data: [str]) -> (np.array, np.array):

        x_data_unified = unify_data(x_data, self.length)

        x_data_windowed = []
        [x_data_windowed.extend(create_timeseries(x, self.window)) for x in x_data_unified]
        y_data = create_y_for(x_data_windowed)

        x_data_encoded = np.array(encode_data(x_data_windowed, self.char2int))
        y_data_encoded = np.array(encode_data(y_data, self.char2int))

        timeseries_num = x_data_encoded.shape[0]
        x_data_encoded_reshaped = x_data_encoded.reshape((timeseries_num, 1, -1))
        y_data_encoded_reshaped = y_data_encoded.reshape((timeseries_num, -1))

        return x_data_encoded_reshaped, y_data_encoded_reshaped
