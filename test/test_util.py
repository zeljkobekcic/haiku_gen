import haiku_gen.util as u
import numpy as np


def test_encode_datapoint():

    data = ['abc', ' ', 'def']
    mapping = {'abc': 0, ' ': 1, 'def': 2}
    output = u.encode_datapoint(data, mapping)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert (output == expected).all()


def test_encode_data():

    data = [['abc', ' ', 'def'], ['.', 'hey', 'def']]
    mapping = {'abc': 0, ' ': 1, 'def': 2, '.': 3, 'hey': 4}
    output = u.encode_data(data, mapping)

    expected = [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]]]

    expected_as_array = np.array(expected)
    assert (output == expected_as_array).all()
