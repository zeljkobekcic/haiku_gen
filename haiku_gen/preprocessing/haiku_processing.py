import haiku_gen.preprocessing.util as u
import haiku_gen.preprocessing.multiprocessing_util as mp_u
import preprocessing.preprocessing_argumentparser as prep_arg

method_function = {'char': u.split_by_characters,
                   'syllable': u.split_by_syllables,
                   'word': u.split_by_words}


def main():

    arguments = prep_arg.PreprocessingArgumentParser()
    input_path = arguments.get_input()
    output_path = arguments.get_output()
    method = arguments.get_method()

    data = u.load_available_data(input_path)

    filter_pipeline = [u.remove_special_characters,
                       u.string_to_lower,
                       u.too_short_except_none,
                       u.verify_haiku_except_none]

    validated = mp_u.multiprocess_data(data, filter_pipeline)
    only_valid_haikus = [d for d, v in zip(data, validated) if v is not None]

    processing_pipeline = [u.whitespace_character_groups,
                           method_function[method]]

    processed = mp_u.multiprocess_data(only_valid_haikus, processing_pipeline)

    train, test = u.create_train_test_data(processed)

    u.save_to(f'{output_path}/train_{method}.pkl', train)
    u.save_to(f'{output_path}/test_{method}.pkl', test)


if __name__ == "__main__":
    main()

