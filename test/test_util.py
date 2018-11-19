import haiku_gen.util as u
import numpy as np


def test_encode_data():

    test_data = ['abc']
    alphabet = 'abcd'
    mapping = {letter: number for letter, number in zip(alphabet, range(len(alphabet)))}

    output = u.encode_data(test_data, mapping)[0]
    expected = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

    assert np.array_equal(output, expected)

    test_data = ['abc', 'cda']
    output = u.encode_data(test_data, mapping)
    expected = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]]),
                np.array([[0,0,1,0], [0,0,0,1], [1,0,0,0]])]

    for o, e in zip(output, expected):
        assert np.array_equal(o, e)


def test_character_set_on_data():

    test_data = ['abc', 'def', 'ghi', 'abci', 'gha']
    output = u.character_set_on_data(test_data)
    expected = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'}

    assert output == expected


def test_mapping_char_to_int():
    test_data = ['abc', 'def']
    validation_data = u.character_set_on_data(test_data)

    output_char_2_int, output_int_2_char = u.mapping_char_int(test_data)

    for c in validation_data:
        assert c == output_int_2_char[output_char_2_int[c]]

