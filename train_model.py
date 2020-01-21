from keras.layers import ConvLSTM2D, Conv3D
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras import backend as K
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sherpa
import argparse

from utils import cache_daily_stacks, load_cache, daily_generator, make_labels, normalize


def build_model(input_shape, parameters):
    n_rows, n_cols, _ = input_shape
    seq = Sequential()
    # Input Layer
    seq.add(ConvLSTM2D(input_shape=(None, n_rows, n_cols, 1),
                       filters=parameters['filters'],
                       kernel_size=parameters['kernel'],
                       dropout=parameters['dropout'],
                       padding='same',
                       return_sequences=True))
    # Layer 1
    seq.add(ConvLSTM2D(filters=parameters['filters'],
                       kernel_size=parameters['kernel'],
                       dropout=parameters['dropout'],
                       padding='same',
                       return_sequences=True
                       ))
    # Layer 2
    seq.add(ConvLSTM2D(filters=parameters['filters'],
                       kernel_size=parameters['kernel'],
                       dropout=parameters['dropout'],
                       padding='same',
                       return_sequences=True
                       ))
    # Layer 3
    seq.add(ConvLSTM2D(filters=parameters['filters'],
                       kernel_size=parameters['kernel'],
                       dropout=parameters['dropout'],
                       padding='same',
                       return_sequences=True
                       ))
    # Output layer
    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    seq.compile(loss='mean_squared_error', optimizer='adadelta')

    return seq


def optimize_model(input_shape, X, Y, study):
    K.set_floatx('float16')

    for trial in study:
        model = build_model(input_shape, trial.parameters)
        epochs = 50
        for i in range(epochs):
            print("Epoch ", i + 1)

            # Cross validation split for each epoch
            x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.2)
            data_generator = daily_generator(x_train, y_train)
            valid_generator = daily_generator(x_validation, y_validation)

            # Train for one epoch and  call back to TensorBoard and Sherpa
            hist = model.fit_generator(generator=data_generator,
                                       steps_per_epoch=len(x_train),
                                       validation_data=valid_generator,
                                       validation_steps=len(x_validation),
                                       use_multiprocessing=True,
                                       initial_epoch=i,
                                       epochs=i + 1,
                                       callbacks=[study.keras_callback(trial, objective_name='val_loss'),
                                                  TensorBoard(log_dir="scripts/logs")])
            print(hist.history)
        study.finalize(trial)


def prep_data(data):

    data = normalize(data, x_min=0, x_max=1000)

    # Get frame shape for our data
    shape = data[0][0].shape

    # Our targets y are simply the shifted frames from X (i.e., each frames target is the frame 2 ahead of itself)
    X, Y = make_labels(data, shift_factor=2)

    # Shifting the data sometimes leaves us with days with no frames, remove these
    X = [dat for dat in X if dat.shape[0] != 0]
    Y = [dat for dat in Y if dat.shape[0] != 0]

    return shape, X, Y


def build_cache(ghi_log, cache_dir):
    # Grid Square for Big Island
    big_east = (19, 20), (-155.5, -154.5)
    big_west = (19, 20), (-156.5, -155.5)

    # Grid Square for Kauai
    kauai = (21.5, 22.48), (-160, -159)

    # Grid Square for Maui
    maui = (20, 20.98), (-156.5, -155.5)

    # Grid Square for Molokai
    molokai = (20.5, 21.48), (-157.5, -156.51)

    # Grid Square for Niihau
    niihau = (21.5, 22.48), (-161, -160.01)

    # Get grid-square around Oahu
    oahu = (21, 21.98), (-158.5, -157.5)

    bounds = [kauai, molokai, big_east, big_west, maui, niihau, oahu]

    islands = ["kauai", "molokai", "big_east", "big_west", "maui", "niihau", "oahu"]

    # Send our tasks to the process pool, as they complete append their results to data
    with Pool(processes=6) as pool:

        print("Caching!!!")
        results = [pool.apply_async(cache_daily_stacks, args=(ghi_log, cache_dir, isle, pos[0], pos[1],))
                   for isle, pos in zip(islands, bounds)]

        for r in results:
            print("Completed Cacheing: ", r.get())


if __name__ == "__main__":
    '''
    islands = ["kauai", "molokai", "big_east", "big_west", "maui", "niihau"]

    data = load_cache(islands, cache_dir="tmp")
    '''

    data = load_cache(["oahu"], cache_dir="tmp")

    shape, X, Y = prep_data(data)

    x, x_test, y, y_test = train_test_split(X, Y, test_size=0.4)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    model = load_model("model.h5")
    model.summary()

    data_gen = daily_generator(x_train, y_train)
    val_gen = daily_generator(x_val, y_val)
    model.fit_generator(generator=data_gen,
                        validation_data=val_gen,
                        epochs=10,
                        steps_per_epoch=len(x_train),
                        validation_steps=len(x_val)
                        )

    model.save("model_01.h5")










