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


def test_character_set_on_data():

    data = [['abc', ' ', 'def'], ['.', 'hey', 'def']]
    output = u.set_of_data_elements(data)
    expected = {'abc', ' ', 'def', '.', 'hey'}

    assert output == expected


def test_mapping_char_int():

    data = [['abc', 'def'], ['abc'], ['def']]
    output = u.mapping_char_int(data)

    #
    expected_1 = ({'abc': 0, 'def': 1}, {0: 'abc', 1: 'def'})
    expected_2 = ({'abc': 1, 'def': 0}, {1: 'abc', 0: 'def'})

    fits_1 = (output[0] == expected_1[0]) and (output[1] == expected_1[1])
    fits_2 = (output[0] == expected_2[0]) and (output[1] == expected_2[1])

    assert fits_1 or fits_2


def test_create_timecell():

    data = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    output = u.create_timecell(data, offset=0, frame=3)
    expected = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]])

    assert np.array_equal(output, expected)


def test_create_timeseries():
    data = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    output = u.create_timeseries(data, frame=3)
    expected = np.array([[[0, 1, 0], [1, 0, 0], [1, 0, 0]],
                         [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
                         [[1, 0, 0], [0, 0, 1], [0, 0, 1]]])

    assert np.array_equal(output, expected)


def test_create_y_for_timeseries():

    data = np.array([[[0, 1, 0], [1, 0, 0], [1, 0, 0]],
                     [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
                     [[1, 0, 0], [0, 0, 1], [0, 0, 1]]])
    output = u.create_y_for_timeseries(data, [1, 1, 1])
    expected = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])

    assert np.array_equal(output, expected)


def test_fill_na():

    data = np.array([[1, 2, 3, 4],
                     [1]])
    output = u.fill_na(data, 0)
    expected = np.array([[1, 2, 3, 4],
                         [1, 0, 0, 0]])

    assert np.array_equal(output, expected)

    data = np.array([[1, 2, 3, 4],
                     [1]])
    length = 7
    output = u.fill_na(data=data, fill=0, length=length)
    expected = np.array([[1, 2, 3, 4, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0]])

    assert np.array_equal(output, expected)


def test_add_start_end():

    data = np.array([[1, 2, 3],
                     [3]])

    output = u.add_start_end(data, left=-1, right=-1)
    expected = np.array([np.array([-1, 1,  2, 3, -1]),
                         np.array([-1, 3, -1])])

    for v1, v2 in zip(output, expected):
        assert np.array_equal(v1, v2)

    data = np.array([['start', 'here'],
                     ['test'],
                     ['hello', 'world', '!']])

    output = u.add_start_end(data, left='START', right='END')
    expected = np.array([['START', 'start', 'here', 'END'],
                         ['START', 'test', 'END'],
                         ['START', 'hello', 'world', '!', 'END']])

    for v1, v2 in zip(output, expected):
        assert np.array_equal(v1, v2)

