from argparse import ArgumentParser
from os.path import expanduser


class PreprocessingArgumentParser(object):

    def __init__(self):
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--output', type=str, metavar='OUT_DIR',
                                     required=True,
                                     help='Absolute path to the output '
                                          'directory. Will be created if not '
                                          'existing')
        argument_parser.add_argument('--input', type=str, metavar='INP_DIR',
                                     required=True,
                                     help='Absolute path to the input '
                                          'directory with the raw data.')
        argument_parser.add_argument('--method', type=str, metavar='METHOD',
                                     required=True,
                                     choices=['char', 'syllable', 'word'],
                                     help='Method how the data will be stored '
                                          'after the preprocessing. It will be '
                                          'stored as a list of characters if '
                                          '\"char\" has been set for this '
                                          'parameter, a list of syllables of '
                                          'the words if the parameter has been '
                                          'set to \"syllable\" and a list of '
                                          'words if the parameter has been set '
                                          'to \"word\".')

        parser = argument_parser.parse_args()

        self.argument_parser = argument_parser
        self.parser = parser

    def get_input(self) -> str:
        input = expanduser(self.parser.input)
        return input

    def get_output(self) -> str:
        output = expanduser(self.parser.output)
        return output

    def get_method(self) -> str:
        method = self.parser.method
        return method
