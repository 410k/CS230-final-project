import argparse
import tensorflow as tf
import os
import numpy as np
from model import MidiNet

import scipy.io
from data_util import load_data


parser = argparse.ArgumentParser(description='How to run the model')
parser.add_argument("-c", "--current_run", required=True,
                    type=str,
                    help="The name of the model which will also be the name of the session's folder")
# network architecture options
parser.add_argument("-hu", "--hidden_units",
                    type=int, default=256,
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
                    type=int, default=44100,
                    help="The sampling frequency (Hz) of the audio input")
parser.add_argument("-tw", "--time_window",
                    type=int, default=50,
                    help="The duration (ms) of each time window")
# training options
parser.add_argument("-ld", "--loss_domain",
                    choices=['time', 'frequency'], default='time',
                    help="The domain in which the loss function is calculated")
parser.add_argument("-bs", "--batch_size",
                    type=int, default=16,
                    help="The number of examples in each mini batch")
parser.add_argument("-rb", "--rebatch",
                    action='store_true',
                    help="Rebatch the data to smaller segments")
parser.add_argument("-rs", "--rebatch_size",
                    type=int, default=200,
                    help="Rebatch the data to 200 time window segments")
parser.add_argument("-lr", "--learning_rate",
                    type=float, default=0.001,
                    help="The learning rate of the RNN")
parser.add_argument("-e", "--epochs",
                    type=int, default=301,
                    help="The total number of epochs to train the RNN for")
parser.add_argument("-ste", "--starting_epoch",
                    type=int, default=0,
                    help="The starting epoch to train the RNN on")
parser.add_argument("-sae", "--save_epoch",
                    type=int, default=10,
                    help="The epoch interval to save the RNN model")
parser.add_argument("-vae", "--validate_epoch",
                    type=int, default=10,
                    help="The epoch interval to validate the RNN model")
parser.add_argument("-eve", "--evaluate_epoch",
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

print('current_run =',        args.current_run)
print('hidden_units =',       args.hidden_units)
print('layers =',             args.layers)
print('unidirectional =',     args.unidirectional)
print('cell_type =',          args.cell_type)
print()
print('sampling_frequency =', args.sampling_frequency)
print('time_window =',        args.time_window)
print()
print('loss_domain =',        args.loss_domain)
print('batch_size =',         args.batch_size)
print('rebatch =',            args.rebatch)
print('rebatch_size =',       args.rebatch_size)
print('learning_rate =',      args.learning_rate)
print('epochs =',             args.epochs)
print('starting_epoch =',     args.starting_epoch)
print('save_epoch =',         args.save_epoch)
print('validate_epoch =',     args.validate_epoch)
print('evaluate_epoch =',     args.evaluate_epoch)
print()
print('load_model =',         args.load_model)
print('load_last =',          args.load_last)
print('data_dir =',           args.data_dir)
print('runs_dir =',           args.runs_dir)


def setup_dir():
    print('[*] Setting up directory...', flush=True)
    main_path = args.runs_dir
    current_run = os.path.join(main_path, args.current_run)
    # data
    data_path = args.data_dir
    train_path = os.path.join(data_path, 'train')
    train_dev_path = os.path.join(data_path, 'train_dev')
    # model
    model_path = os.path.join(current_run, 'model')
    logs_path = os.path.join(current_run, 'logs')
    png_path = os.path.join(current_run, 'png')
    pred_path = os.path.join(current_run, 'predictions')

    dirs = {'main_path': main_path,
            'current_run': current_run,
            'train_path': train_path,
            'train_dev_path': train_dev_path,
            'model_path': model_path,
            'logs_path': logs_path,
            'png_path': png_path,
            'pred_path': pred_path}
    # create the directories if they don't already exist
    for dir_name, dir_path in dirs.items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    return dirs


def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    dirs = setup_dir()

    data = {}
    data['classical'] = {}
    data['classical']['X'], data['classical']['Y'], _ = load_data(dirs['train_dev_path'])
    input_size = data['classical']['X'][0].shape[1]
    output_size = data['classical']['Y'][0].shape[1]

    network  = MidiNet(dirs, 
                       input_size=input_size, 
                       output_size=output_size, 
                       num_hidden_units=args.hidden_units,  
                       num_layers=args.layers,
                       unidirectional_flag=args.unidirectional)
    network.prepare_model()

    if args.load_model:
        # assumes that model name is [name]-[e][epoch_number]
        loaded_epoch = args.load_model.split('.')[0]
        loaded_epoch = loaded_epoch.split('-')[-1]
        loaded_epoch = loaded_epoch[1:]
        print("[*] Loading " + args.load_model + " and continuing from " + loaded_epoch, flush=True)
        loaded_epoch = int(loaded_epoch)
        model = args.load_model
        starting_epoch = loaded_epoch+1
    elif args.load_last:
        # list all the .ckpt files in a tuple (epoch, model_name)
        tree = os.listdir(dirs["model_path"])
        tree.remove('checkpoint')
        files = [(int(file.split('.')[0].split('-')[-1][1:]), file.split('.')[0]) for file in tree]
        # find the properties of the last checkpoint
        files.sort(key = lambda t: t[0])
        target_file = files[-1]
        loaded_epoch = target_file[0]
        model_name = target_file[1]
        model = model_name + ".ckpt"
        print("[*] Loading " + model + " and continuing from epoch " + str(loaded_epoch), flush=True)
        starting_epoch = loaded_epoch+1
    else:
        model = None
        starting_epoch = args.starting_epoch

    if args.mode == 'train':
        data = {}
        data['classical'] = {}
        data['classical']['X'], data['classical']['Y'], _ = load_data(dirs['train_path'])
        input_size = data['classical']['X'][0].shape[1]
        output_size = data['classical']['Y'][0].shape[1]
        #
        network.train(data, 
                      model=model,
                      starting_epoch=starting_epoch,
                      batch_size=args.batch_size,
                      rebatch_flag=args.rebatch,
                      rebatch_size=args.rebatch_size,
                      learning_rate=args.learning_rate,
                      epochs=args.epochs, 
                      save_epoch=args.save_epoch, 
                      val_epoch=args.validate_epoch,
                      eval_epoch=args.evaluate_epoch)
    else:
        network.load(model)

        data = {}
        data['inputs'], data['outputs'], filenames = load_data(dirs['train_dev_path'])
        #
        # prev_filename = filenames[0]

        for ind, filename in enumerate(filenames):
            tmp_filename = filename.split('.')[0] + '_' + str(ind)
            single_input = data['inputs'][ind]
            single_output = data['outputs'][ind]
            if len(single_input.shape) == 2:
                single_input = np.expand_dims(single_input, axis=0)
                single_output = np.expand_dims(single_output, axis=0)

            loss, model_output, _ = network.predict(single_input, single_output)

            # create a figure
            network.plot_evaluation(loaded_epoch, tmp_filename, single_input, single_output, model_output)
        
            # save the data
            save_dict = {'model_output': model_output,
                         'true_output': single_output,
                         'input': single_input}
            filepath = os.path.join(dirs['pred_path'], tmp_filename.split('.')[0] + "-e%d" % (loaded_epoch)+".mat")
            scipy.io.savemat(filepath, save_dict)


if __name__ == '__main__':
    main()
