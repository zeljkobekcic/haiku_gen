import pickle as p
import numpy as np
import hashlib as h
import datetime as dt
import warnings
import os
import keras

from functools import reduce
from keras.utils import to_categorical


def load_data_from(path):
    """Loads data from the given path. Basically just a nice little function
    to easier use p.load """
    with open(path, 'rb') as infile:
        return p.load(infile)


def load_data(method):
    """Loads data from the default path 'data/train_{method}. This is the same
    as load_data_from(data/train_{method}.pkl)."""
    local_path = 'data/train_{method}.pkl'.format(method=method)
    return load_data_from(local_path)


def load_test_data(method):
    """Loads data from the default path 'data/train_{method}. This is the same
    as load_data_from(data/train_{method}.pkl)."""
    local_path = 'data/test_{method}.pkl'.format(method=method)
    return load_data_from(local_path)


def encode_datapoint(datapoint: list, mapping: dict) -> list:
    """Encodes a single datapoint with the given mapping.
    :param datapoint: A single datapoint
    :param mapping: A dictionary which maps the values of the elements in data
                    to integers so that these can be mapped to an one-hot
                    encoding.
    For example if we had the string: 'hello' and the mapping
    m = {'h': 0, 'e':1, 'l': 2, 'o', 3} then this function will return the
    following:

    [np.array([1 ,0, 0, 0]), # h
     np.array([0, 1, 0, 0]), # e
     np.array([0, 0, 1, 0]), # l
     np.array([0, 0, 1, 0]), # l
     np.array([0, 0, 0, 1])] # o
    """
    return to_categorical([mapping[d] for d in datapoint],
                          len(mapping),
                          dtype='int')


def encode_data(data: [list], mapping: dict) -> [list]:
    """Applies encode_datapoint for a collection of datapoints.

    :param data: A list of datapoints which will be passed one by one to
                 encode_datapoint
    :param mapping: A dictionary which maps the values of the elements in data
                    to integers so that these can be mapped to an one-hot
                    encoding.
    :returns A numpy array of the lists which represent the encoded values."""
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
    :return: Two dictionaries which store the mapping from our
             data input X -> encoded data X~ and vice versa.
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


def add_start_end(data: np.array, left, right) -> np.array:
    """Adds an start and end sequence element to the datapoints.

    :param data:
    :param left, right
    :return
    """
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
    """padds the data so that every element has the same length.

    :param data: The input data, should be an iterable collection.
    :param left, right: Elements which will be added to the datapoints to
                              mark the start, end of an sequence.
    :param fill: Filling elements which are shorter than others with this
                 value.

    :return The data with set start and stop sequence elements where every
            element has the same length
    """
    start_end = add_start_end(data, left, right)
    padded = fill_na(start_end, fill, length=length)
    return padded


def create_y_for_timeseries(datapoint: [list], end_seq: list) -> [list]:
    """Takes a datapoints of sequence elements and returns a list of elements
    which contain the next sequence element from the sequencep.

    :param datapoint: The input data.
    :param end_seq: The last sequence element, because the last element in the
                     sequence has no following element.
    :return A corresponding y vector for the input data x.
    """

    try:
        y = [d[-1] for d in datapoint[1:]]
    except IndexError:
        print(datapoint)

    y.append(end_seq)
    return y


def save_model(model: keras.Sequential, naming_params: {str: str},
               directory='model/'):
    """Saves a given model on the disk as a hdf5 file.

    :param model: A keras.Sequential model
    :param naming_params: A dictionary with keys and values for specific
                          properties.
    :param directory: The path to the targeted directory.
    """
    os.makedirs('model', exist_ok=True)

    naming_params['time'] = dt.datetime.now().strftime("%Y:%m:%d::%H:%M:%S")
    naming_params['hash'] = h.sha1(model.to_json().encode()).hexdigest()

    def key_and_value_to_str(key):
        return "{key}={value}".format(key=key, value=naming_params[key])

    name_parts = map(key_and_value_to_str, sorted(naming_params))
    model_name = '-'.join(name_parts) + ".hdf5"
    model.save(directory + '/' + model_name)


def load_model_from_hash(model_hash: str, directory: str) -> keras.Sequential:
    hash_field = 'hash=' + model_hash
    files = filter(lambda x: x.endswith('.hdf5') and hash_field in x, directory)

    if len(files) > 1:
        warnings.warn("More than one model starting with the provided hash:\n" +
                      "Using the first element in the following operations.\n" +
                      "Provided hash: {}\n".format(model_hash) +
                      "Direcotory: {}\n\t".format(directory) +
                      "\n\t".join(files))

    return keras.models.load_model(files[0])


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
