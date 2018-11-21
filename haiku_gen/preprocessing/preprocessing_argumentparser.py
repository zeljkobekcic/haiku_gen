from argparse import ArgumentParser
from os.path import expanduser


class PreprocessingArgumentParser(object):

    def __init__(self):
        argument_parser = ArgumentParser()
        argument_parser.add_argument('--output', type=str, metavar='OUT_DIR', required=True,
                                     help='Absolute path to the output directory. Will be created if not existing')
        argument_parser.add_argument('--input', type=str, metavar='INP_DIR', required=True,
                                     help='Absolute path to the input directory with the raw data.')

        parser = argument_parser.parse_args()

        self.argument_parser = argument_parser
        self.parser = parser

    def get_input(self) -> str:
        input = expanduser(self.parser.input)
        return input

    def get_output(self) -> str:
        output = expanduser(self.parser.output)
        return output
