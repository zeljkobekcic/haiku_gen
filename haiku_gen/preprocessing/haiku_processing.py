import haiku_gen.preprocessing.util as u
import haiku_gen.preprocessing.multiprocessing_util as mp_u
import preprocessing.preprocessing_argumentparser


def main():

    arguments = preprocessing.preprocessing_argumentparser.PreprocessingArgumentParser()
    input_path = arguments.get_input()
    output_path = arguments.get_output()

    data = u.load_available_data(input_path)

    pipeline = [u.remove_special_characters,
                u.string_to_lower,
                u.to_short_and_none,
                u.verify_haiku_except_none]

    validated = mp_u.multiprocess_data(data, pipeline)
    only_valid_haikus = [d for d, v in zip(data, validated) if v is not None]
    train, test = u.create_train_test_data(only_valid_haikus)

    u.save_to(f'{output_path}/train.pkl', train)
    u.save_to(f'{output_path}/test.pkl', test)


if __name__ == "__main__":
    main()

