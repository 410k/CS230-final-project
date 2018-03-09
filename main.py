import argparse
parser = argparse.ArgumentParser(description='How to run the model')
parser.add_argument("-c", "--current_run", required=True,
                    type=str,
                    help="The name of the model which will also be the name of the session's folder")
# network architecture options
parser.add_argument("-hu", "--hidden_units",
                    type=int, default=128,
                    help="The number of hidden units per layer in the RNN")
parser.add_argument("-l", "--layers",
                    type=int, default=2,
                    help="The number of layers in the RNN")
parser.add_argument("-uni", "--unidirectional",
                    action='store_true',
                    help="Use a unidirectional RNN network instead of a bidirectional network")
parser.add_argument("-ct", "--cell_type",
                    choices=['GRU', 'LSTM'], default='GRU',
                    help="Memory cell type to use in the RNN")
# input data options
parser.add_argument("-sf", "--sampling_frequency",
                    type=int, default=8000,
                    help="The sampling frequency (Hz) of the audio input")
parser.add_argument("-tw", "--time_window_duration",
                    type=float, default=0.05,
                    help="The duration (s) of each time window")
parser.add_argument("-ed", "--example_duration",
                    type=float, default=4.0,
                    help="The duration (s) of each example")
# training options
parser.add_argument("-g", "--gpus",
                    type=int, default=0,
                    help="The number of GPUs to use")
parser.add_argument("-ld", "--loss_domain",
                    choices=['time', 'frequency'], default='time',
                    help="The domain in which the loss function is calculated")
parser.add_argument("-bs", "--batch_size",
                    type=int, default=8,
                    help="The number of examples in each mini batch")
parser.add_argument("-lr", "--learning_rate",
                    type=float, default=0.001,
                    help="The learning rate of the RNN")
parser.add_argument("-e", "--epochs",
                    type=int, default=301,
                    help="The total number of epochs to train the RNN for")
parser.add_argument("-ste", "--starting_epoch",
                    type=int, default=0,
                    help="The starting epoch to train the RNN on")
parser.add_argument("-esi", "--epoch_save_interval",
                    type=int, default=10,
                    help="The epoch interval to save the RNN model")
parser.add_argument("-evi", "--epoch_val_interval",
                    type=int, default=10,
                    help="The epoch interval to validate the RNN model")
parser.add_argument("-eei", "--epoch_eval_interval",
                    type=int, default=10,
                    help="The epoch interval to evaluate the RNN model")
# other options
parser.add_argument("-lm", "--load_model",
                    type=str, default=None,
                    help="Folder name of model to load")
parser.add_argument("-ll", "--load_last",
                    action='store_true',
                    help="Start from last epoch")
parser.add_argument("-m", "--mode",
                    choices=['train', 'predict'], default='train',
                    help="Mode to operate model in")
# file system options
parser.add_argument("--data_dir",
                    type=str, default="./data",
                    help="Directory of datasets")
parser.add_argument("--runs_dir",
                    type=str, default="./runs",
                    help="The name of the model which will also be the name of the session folder")
args = parser.parse_args()

print()
print('current_run =',        args.current_run)
print('hidden_units =',       args.hidden_units)
print('layers =',             args.layers)
print('unidirectional =',     args.unidirectional)
print('cell_type =',          args.cell_type)
print()
print('sampling_frequency =', args.sampling_frequency)
print('time_window_duration =', args.time_window_duration)
print('example_duration =',   args.example_duration)
print()
print('loss_domain =',        args.loss_domain)
print('batch_size =',         args.batch_size)
print('learning_rate =',      args.learning_rate)
print('epochs =',             args.epochs)
print('starting_epoch =',     args.starting_epoch)
print('epoch_save_interval =',args.epoch_save_interval)
print('epoch_val_interval =', args.epoch_val_interval)
print('epoch_eval_interval =',args.epoch_eval_interval)
print()
print('load_model =',         args.load_model)
print('load_last =',          args.load_last)
print('data_dir =',           args.data_dir)
print('runs_dir =',           args.runs_dir)
print()

import os
import numpy as np
import tensorflow as tf
from MidiNet import MidiNet
from sklearn.model_selection import train_test_split

import scipy.io
from util import setup_dirs, load_data, save_predictions
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model


def main():
    dirs = setup_dirs(args)

    example_duration = args.example_duration
    time_window_duration = args.time_window_duration
    sampling_frequency = args.sampling_frequency

    num_hidden_units = args.hidden_units
    num_layers = args.layers
    unidirectional_flag = args.unidirectional
    cell_type = args.cell_type
    batch_size = args.batch_size

    num_epochs = args.epochs
    epoch_save_interval = args.epoch_save_interval

    gpus = args.gpus

    #
    if args.load_model:
        # assumes that model name is [name]-[e][epoch_number]-[other_stuff]
        model_filename = args.load_model
        model_epoch_str = model_filename.split('.')[0].split('-')[1]
        model_epoch = model_epoch_str[1:]
        print("[*] Loading " + args.load_model + " and continuing from epoch " + model_epoch, flush=True)
        model_path = os.path.join(dirs['model_path'], model_filename)
        model = load_model(model_path)
        starting_epoch = int(model_epoch)+1
    elif args.load_last:
        # list all the .ckpt files in a tuple (epoch, model_name)
        tree = os.listdir(dirs["model_path"])
        files = [(int(file.split('.')[0].split('-')[1][1:]), file.split('.')[0]) for file in tree]
        # find the properties of the last checkpoint
        files.sort(key = lambda t: t[0])
        target_file = files[-1]
        model_epoch = target_file[0]
        model_name = target_file[1]
        model_filename = model_name + ".hdf5"
        print("[*] Loading " + model_filename + " and continuing from epoch " + str(model_epoch), flush=True)
        model_path = os.path.join(dirs['model_path'], model_filename)
        model = load_model(model_path)
        starting_epoch = int(model_epoch)+1
    else:
        starting_epoch = 0

    #
    if args.mode == 'train':
        # load data
        train_path = dirs['train_path']
        X_data, Y_data, filenames = load_data(train_path, example_duration, time_window_duration, sampling_frequency)
        X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.1)
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = (Y_train.shape[1], Y_train.shape[2])

        # create & compile model
        if not 'model' in vars():
            with tf.device('/cpu:0'):
                model = MidiNet(input_shape, output_shape, num_hidden_units, num_layers, 
                    unidirectional_flag, cell_type)
            if gpus >= 2:
                model = multi_gpu_model(model, gpus=gpus)
            model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())

        # train the model & run a checkpoint callback
        checkpoint_filename = 'model-e{epoch:03d}-loss{loss:.4f}.hdf5'
        checkpoint_filepath = os.path.join(dirs['model_path'], checkpoint_filename)
        checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='loss', verbose=1, period=epoch_save_interval)
        csv_filename = 'training_log.csv'
        csv_filepath = os.path.join(dirs['current_run'], csv_filename)
        csv_logger = CSVLogger(csv_filepath)
        callbacks_list = [checkpoint, csv_logger]
        history_callback = model.fit(X_train, Y_train, epochs=num_epochs+starting_epoch, 
            initial_epoch=starting_epoch, batch_size=batch_size, callbacks=callbacks_list)

        # save the loss history
        loss_history = history_callback.history["loss"]
        save_dict = {'loss_history': loss_history}
        filepath = os.path.join(dirs['current_run'], "loss_history.mat")
        scipy.io.savemat(filepath, save_dict)

        # save the final model
        last_epoch = history_callback.epoch[-1]
        filename = 'model-e' + str(last_epoch) + '.hdf5'
        filepath = os.path.join(dirs['model_path'], filename)
        model.save(filepath)

        # evaluate model on training and test data
        Y_test_pred = model.predict(X_test, batch_size=batch_size)
        Y_train_pred = model.predict(X_train, batch_size=batch_size)

        # save predictions
        save_predictions(dirs['pred_path'], 'test', X_test, Y_test, Y_test_pred)
        save_predictions(dirs['pred_path'], 'train', X_train, Y_train, Y_train_pred)
    # else:
    #     network.load(model)

    #     data = {}
    #     data['inputs'], data['outputs'], filenames = load_data(dirs['train_dev_path'])
    #     #
    #     # prev_filename = filenames[0]

    #     for ind, filename in enumerate(filenames):
    #         tmp_filename = filename.split('.')[0] + '_' + str(ind)
    #         single_input = data['inputs'][ind]
    #         single_output = data['outputs'][ind]
    #         if len(single_input.shape) == 2:
    #             single_input = np.expand_dims(single_input, axis=0)
    #             single_output = np.expand_dims(single_output, axis=0)

    #         loss, model_output, _ = network.predict(single_input, single_output)

    #         # create a figure
    #         network.plot_evaluation(loaded_epoch, tmp_filename, single_input, single_output, model_output)
        
    #         # save the data
    #         save_dict = {'model_output': model_output,
    #                      'true_output': single_output,
    #                      'input': single_input}
    #         filepath = os.path.join(dirs['pred_path'], tmp_filename.split('.')[0] + "-e%d" % (loaded_epoch)+".mat")
    #         scipy.io.savemat(filepath, save_dict)


if __name__ == '__main__':
    main()