from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.layers import LeakyReLU

import haiku_gen.util as u
import numpy as np
import numpy.random as rnd


def generator(data: np.array, ts_enc: u.TimeseriesEncoder, batch_size: int, random_seed=123):

    counter = 0
    rnd.seed(random_seed)
    index = rnd.permutation(data.shape[0])

    while True:
        batch_start = counter * batch_size
        batch_end = (counter + 1) * batch_size
        batch_index = index[batch_start: batch_end]
        batch = data[batch_index]
        x_data, y_data = ts_enc.transform(batch)
        yield x_data, y_data

        counter += 1

        if np.ceil(len(data) / batch_size) <= counter:
            counter = 0
            index = rnd.permutation(data.shape[0])


def tensor_shape(ts_enc: u.TimeseriesEncoder, batch) -> (int, int, int):
    return ts_enc.transform(batch)[0].shape


def encoded_shape(ts_enc: u.TimeseriesEncoder) -> int:
    return len(ts_enc.char2int)


def main():
    d = u.load_data('char')

    batch_size = 10
    window = 5
    num_elements = 10
    epochs = 10
    steps_per_epoch = num_elements // batch_size

    d_batch = d[:batch_size]
    ts_enc = u.TimeseriesEncoder(data=d, window=window, right='<END>',
                                 left='<START>', fill='<END>')
    ts_enc.save_to_file()
    batch_shape = tensor_shape(ts_enc, d_batch)
    output_shape = encoded_shape(ts_enc)

    print(batch_shape)
    print(output_shape)

    model = Sequential()
    model.add(SimpleRNN(units=200, batch_input_shape=batch_shape))
    model.add(LeakyReLU())
    model.add(Dense(units=output_shape, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    g = generator(data=d[:num_elements], ts_enc=ts_enc, batch_size=batch_size)
    model.fit_generator(g, steps_per_epoch=steps_per_epoch, epochs=epochs)

    naming_params = {'arch': 'SimpleRNN',
                     'epochs': epochs,
                     'num_elements': num_elements,
                     'window': window,
                     'batch_size': batch_size}

    u.save_model(model, naming_params)


if __name__ == '__main__':
    main()