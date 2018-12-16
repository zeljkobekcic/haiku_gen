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


def set_of_data_elements(data: list) -> dict:
    """
    Creates a set of characters over a collection of strings.
    :param data: A collection of strings.
    :return: The used characters in the collection of strings.
    """
    return reduce(lambda x, y: x | y, [set(x_i) for x_i in data])


def mapping_char_int(data: [list]) -> (dict, dict):
    """
    Creates two mappings. One from characters to numbers and one from numbers to characters.
    :param data: A collection os strings.
    :return: Two dictionaries.
    """
    char_set = set_of_data_elements(data)
    char_2_int = {c: i for i, c in enumerate(char_set)}
    int_2_char = {i: c for i, c in enumerate(char_set)}

    return char_2_int, int_2_char


def create_timecell(datapoint: np.array, offset: int, frame: int) -> np.array:
    """
    Helper function to create timecells for the recurrent neural networks.

    :param datapoint:
    :param offset:
    :param frame:
    :return: A timecell from a datapoint.
    """
    timecell = datapoint[offset:offset + frame]
    return timecell


def create_timeseries(datapoint: np.array, frame: int) -> np.array:
    """
    A helper function to create timeseries from datapoints.

    :param datapoint:
    :param frame:
    :return: A timeseries from a datapoint
    """

    def timecell(offset):
        return create_timecell(datapoint, offset, frame)

    timecell_num = len(datapoint) - frame + 1
    timeseries = np.array([timecell(i) for i in range(timecell_num)])
    return timeseries


def add_start_end(data, left, right) -> np.array:
    x = np.zeros_like(data, dtype=object)
    for i, v in enumerate(data):
        x[i] = np.pad(array=np.array(v, dtype=object),
                      pad_width=(1, 1),
                      mode='constant',
                      constant_values=(left, right))
    return x


def fill_na(data, fill, length=None):
    """
    Taken from: https://stackoverflow.com/a/32043366/3446982 and slightly
    modified
    """
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    if length is None:
        length = lens.max()

    # Mask of valid places in each row
    mask = np.arange(length) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.full(mask.shape, fill, dtype=object)
    out[mask] = np.concatenate(data)
    return out


def unify_data(data, left, right, fill, length=None):
    start_end = add_start_end(data, left, right)
    padded = fill_na(start_end, fill, length=length)
    return padded


def create_y_for_timeseries(datapoint: [list], end_seq: list) -> [list]:
    try:
        y = [d[-1] for d in datapoint[1:]]
    except IndexError:
        print(datapoint)
    y.append(end_seq)
    return y


def save_model(model, naming_params):
    import os
    os.makedirs('model', exist_ok=True)
    model_name = 'arch={arch}-epochs={epochs}-num_elements={num_elements}-window={window}-batch_size={batch_size}.hdf5'
    model.save('model/' + model_name.format(**naming_params))


class TimeseriesEncoder:

    def __init__(self, data, left, right, fill, window=5):

        self.left = left
        self.right = right
        self.fill = fill

        data_unified = unify_data(data=data, left=left, right=right, fill=fill)

        self.char2int, self.int2char = mapping_char_int(data_unified)

        try:
            self.char2int[left]
            self.char2int[right]
            self.char2int[fill]
        except KeyError:
            print("NO")
            print(data_unified.dtype)

        self.length = data_unified.shape[1]
        self.window = window

    def transform(self, data: [str]) -> (np.array, np.array):
        x_data_unified = unify_data(data=data, fill=self.fill, right=self.right,
                                    left=self.left, length=self.length)

        x_data_windowed = [timecell for x in x_data_unified
                           for timecell in create_timeseries(x, self.window)]

        y_data = create_y_for_timeseries(x_data_windowed, self.fill)

        x_data_encoded = np.array(encode_data(x_data_windowed, self.char2int))
        y_data_encoded = np.array(encode_datapoint(y_data, self.char2int))

        timeseries_num = x_data_encoded.shape[0]
        x_data_encoded_reshaped = x_data_encoded.reshape((timeseries_num, 1, -1))
        y_data_encoded_reshaped = y_data_encoded.reshape((timeseries_num, -1))

        return x_data_encoded_reshaped, y_data_encoded_reshaped
