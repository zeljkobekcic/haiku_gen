import haiku_gen.preprocessing.util as u
import haiku_gen.preprocessing.multiprocessing_util as mp_u

from haiku_gen.preprocessing.util import remove_special_characters, string_to_lower, to_short_and_none, remove_none


def main():

    arguments = u.PreprocessingArgumentParser()
    input_path = arguments.get_input()
    output_path = arguments.get_output()

    data = u.load_available_data(input_path)

    pipeline = [remove_special_characters,
                string_to_lower,
                to_short_and_none]

    mapped = mp_u.multiprocess_data(data, pipeline)
    only_valid_haikus = remove_none(mapped)
    train, test = u.create_train_test_data(only_valid_haikus)

    u.save_to(f'{output_path}/train.pkl', train)
    u.save_to(f'{output_path}/test.pkl', test)


if __name__ == "__main__":
    main()

