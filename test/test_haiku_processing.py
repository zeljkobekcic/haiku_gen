import haiku_gen.preprocessing.util as u


def test_is_too_short():

    input = 'this is definitely not too short'
    output = u.is_too_short(input)
    expected = False

    assert output == expected

    input = 'too short'
    output = u.is_too_short(input)
    expected = True

    assert output == expected


def test_remove_special_characters():

    input = 'there is no : special character #$%^ anymore'
    output = u.remove_special_characters(input)
    expected = 'there is no  special character  anymore'

    assert expected == output


def test_too_short_and_none():

    input = 'this is too short, definitely'
    output = u.too_short_except_none(input)
    assert output is not None

    input = 'too short'
    output = u.too_short_except_none(input)
    assert output is None


def test_remove_none():

    input = [None, 1, None, 2, None, 'hello']
    output = u.remove_none(input)
    expected = [1, 2, 'hello']
    assert output == expected


def test_verify_haiku():

    input = 'A B C D E\nF G H I K L M\nN O P Q R'
    output = u.verify_haiku(input)
    expected = True
    assert output == expected

    input = 'AB C D E\nF G H I K L M\nN O P Q R'
    output = u.verify_haiku(input)
    expected = False
    assert output == expected


def test_count_syllables_for_line():

    input = ['A', 'B', 'C']
    output = u.count_syllables_for_line(input)
    expected = 3

    assert output == expected


def test_count_syllables():

    input = 'some'
    output = u.count_syllables(input)
    expected = 1

    assert output == expected


def test_whitespace_character_groups():

    input = 'This is...some simple! TEST!'
    output = u.whitespace_character_groups(input)
    expected = 'This is ... some simple ! TEST !'
    assert output == expected


def test_split_by_syllables():
    input = 'This is ... some simple ! TEST !'
    output = u.split_by_syllables(input)
    expected = ['This', ' ', 'is', ' ', '...', ' ', 'some', ' ', 'sim', 'ple',
                ' ', '!', ' ', 'TEST', ' ', '!']

    assert output == expected


def test_split_by_words():
    input = 'This is ... some simple ! TEST !'
    output = u.split_by_words(input)
    expected = ['This', ' ', 'is', ' ', '...', ' ', 'some', ' ', 'simple', ' ',
                '!', ' ', 'TEST', ' ', '!']

    assert output == expected
